
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Optional, Tuple, Dict, Any, List
from torch.utils.tensorboard import SummaryWriter
import os
import random
import pandas as pd
import numpy as np
import ast
from pathlib import Path
from dataclasses import dataclass, field


from model_base import MultiTaskMLP
from embedding import getEmbedding
from data_split import stratify_split_indices
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from sklearn.decomposition import PCA

def fix_seed_new(seed: int = 0) -> None:
    """Fix random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_metrics_with_stats(true: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    """Simple metric wrapper for reuse in the trainer."""

    return {
        "mse":mean_squared_error(true, pred),
        "mae": mean_absolute_error(true, pred),
        "r2": r2_score(true, pred),
        "pearsonr":np.corrcoef(true, pred)[0, 1],
        "num_samples": len(true),
    }


@dataclass
class TrainConfigNew:
    run_type: str
    csv_path: str
    model_dir: Path
    peptide_seq_ls_name: str
    peptide_smis_ls_name: str
    main_smis_name: str
    feature_type: List[str]
    activity_columns: List[str]
    split_ratio: float = 0.1
    split_method: str = 'stratify'
    seed: int = 42
    # Auto-compute weights from train-set valid label counts
    task_weight_strategy: str = "inverse_freq"
    batch_size: int = 64
    num_epochs: int = 100
    early_stop_patience: int = 10
    hidden: List[int] = field(default_factory=lambda: [512, 256])
    dropout: float = 0.2
    lr: float = 1e-4
    weight_decay: float = 1e-4


class MultiTaskMLPTrainer:
    def __init__(self, config: TrainConfigNew):
        fix_seed_new(0)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config.model_dir.mkdir(parents=True, exist_ok=True)

        # Task weight tensor at train time (from task_weight_strategy based on train valid counts)
        self.task_weights: Optional[torch.Tensor] = None  # shape [n_tasks]
        self.optimizer: torch.optim.Optimizer = None
        self.scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau = None
        self.criterion: nn.Module = nn.MSELoss()

        self.best_val_loss = float("inf")
        self.best_epoch = 1
        self.writer: SummaryWriter | None = None
        # Per-task label standardization params (computed on train set)
        self.label_mean: np.ndarray | None = None
        self.label_std: np.ndarray | None = None


    def load_data(self):
        df = pd.read_csv(self.config.csv_path)

        # Parse string-formatted lists
        fasta_list = df[self.config.peptide_seq_ls_name].apply(ast.literal_eval).tolist()
        frag_list = df[self.config.peptide_smis_ls_name].apply(ast.literal_eval).tolist()
        smiles_list = df[self.config.main_smis_name].tolist()
        activity_array = df[self.config.activity_columns].values
        print(f"Label range: max={np.nanmax(activity_array)}, min={np.nanmin(activity_array)}")

        # Split indices with stratify_split_indices
        train_idx, val_idx, test_idx = stratify_split_indices(
            activity_array, val_ratio=self.config.split_ratio, test_ratio=self.config.split_ratio, seed=self.config.seed)
        print(f'train_number:{len(train_idx)}, val_number:{len(val_idx)}, test_number:{len(test_idx)}')
        df['TrTe'] = 'Train'
        df.loc[val_idx, 'TrTe'] = 'Validation'
        df.loc[test_idx, 'TrTe'] = 'Test'
        df.to_csv(Path(self.config.model_dir) / 'modeling_data.csv', index=None)

        # Split fasta / frag / activity by index
        fasta_train, frag_train = [fasta_list[i] for i in train_idx], [frag_list[i] for i in train_idx]
        fasta_val, frag_val = [fasta_list[i] for i in val_idx], [frag_list[i] for i in val_idx]
        fasta_test, frag_test = [fasta_list[i] for i in test_idx], [frag_list[i] for i in test_idx]
        smiles_train = [smiles_list[i] for i in train_idx]
        smiles_val = [smiles_list[i] for i in val_idx]
        smiles_test = [smiles_list[i] for i in test_idx]

        activity_train = activity_array[train_idx]
        activity_val = activity_array[val_idx]
        activity_test = activity_array[test_idx]

        # Count valid labels per task on train set and build task weights (for balancing)
        # counts: [n_tasks]
        counts = np.sum(~np.isnan(activity_train), axis=0).astype(np.float32)
        # Avoid division by zero
        safe_counts = np.where(counts <= 0, np.nan, counts)
        strategy = (self.config.task_weight_strategy or "inverse_freq").lower()
        if strategy == "equal":
            w = np.ones_like(counts, dtype=np.float32)
        elif strategy == "inverse_sqrt_freq":
            mean_cnt = np.nanmean(safe_counts)
            w = np.sqrt(mean_cnt / safe_counts)
        else:  # inverse_freq (default)
            mean_cnt = np.nanmean(safe_counts)
            w = mean_cnt / safe_counts
        # If a task has no labels in train, set weight to 0 (excluded from loss)
        w = np.where(np.isfinite(w), w, 0.0).astype(np.float32)

        self.task_weights = torch.tensor(w, dtype=torch.float32, device=self.device)
        print("Train label counts:", dict(zip(self.config.activity_columns, counts.astype(int).tolist())))
        print("Task weights:", dict(zip(self.config.activity_columns, w.tolist())))

        # Z-score standardize each task label using train set
        # [n_train, n_tasks]
        mean = np.nanmean(activity_train, axis=0)
        std = np.nanstd(activity_train, axis=0)
        # Handle zero or NaN variance
        std[std == 0] = 1.0
        std = np.where(np.isnan(std), 1.0, std)
        mean = np.where(np.isnan(mean), 0.0, mean)

        self.label_mean = mean
        self.label_std = std

        def standardize(arr: np.ndarray) -> np.ndarray:
            return (arr - mean) / std

        activity_train = standardize(activity_train)
        activity_val = standardize(activity_val)
        activity_test = standardize(activity_test)

        # Save standardization params to disk for prediction
        scaler_path = Path(self.config.model_dir) / "label_scaler.npz"
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(scaler_path, mean=mean, std=std)

        # Compute embeddings on train set, fit scaler, save to model_dir
        train_dataset = getEmbedding(fasta_train, frag_train, smiles_train, self.config.feature_type, \
                                    run_type="train", activity_array=activity_train, save_path=self.config.model_dir)
        # Transform val / test with the same scaler
        val_dataset = getEmbedding(fasta_val, frag_val, smiles_val, self.config.feature_type, \
                                    run_type="predict", activity_array=activity_val, save_path=self.config.model_dir,)
                                    
        test_dataset = getEmbedding(fasta_test,frag_test, smiles_test, self.config.feature_type, \
                                    run_type="predict", activity_array=activity_test, save_path=self.config.model_dir)
        # # For subsequent AD computation
        self.train_AD = train_dataset.tensors[0].detach().cpu().numpy()
        print(f"AD feature dimensions: {self.train_AD.shape}")


  
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset,  batch_size=self.config.batch_size, shuffle=False)
        
        # Input and output dimensions
        self.input_dim = train_dataset.tensors[0].shape[1]
        self.output_dim = len(self.config.activity_columns)  # 3 tasks
        print(f'Input dim: {self.input_dim}, hidden: {self.config.hidden}, output dim: {self.output_dim}')

    # -------------------------
    # Single-epoch train / validation
    # -------------------------

    def train_one_epoch(self, epoch: int) -> float:
        """Single-epoch training for multi-task regression."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # batch_embeddings: [batch_size, feature_dim]
        # batch_targets: [batch_size, num_targets]
        for batch_embeddings, batch_targets, _ in self.train_loader:
            batch_embeddings = batch_embeddings.to(self.device)
            batch_targets = batch_targets.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(batch_embeddings)  # [batch, num_tasks]

            loss = 0.0
            denom = 0.0

            for task_idx in range(batch_targets.size(1)):
                task_mask = ~torch.isnan(batch_targets[:, task_idx])
                if task_mask.any():
                    w = float(self.task_weights[task_idx].item()) if self.task_weights is not None else 1.0
                    # Skip tasks with zero weight (e.g., no labels in train)
                    if w <= 0:
                        continue
                    loss = loss + w * self.criterion(
                        outputs[:, task_idx][task_mask],
                        batch_targets[:, task_idx][task_mask],
                    )
                    denom += w

            if denom > 0:
                loss = loss / denom
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        # print(f"[Epoch {epoch}] Train Opt Loss: {avg_loss:.4f}")
        return avg_loss


    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, Dict[str, Any]]]:
        """Evaluate: return average loss and per-task metrics."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        task_true = []
        task_pred = []

        for batch_embeddings, batch_targets, _ in dataloader:
            batch_embeddings = batch_embeddings.to(self.device)
            batch_targets = batch_targets.to(self.device)

            outputs = self.model(batch_embeddings)

            batch_loss = 0.0
            denom = 0.0
            # Only backprop on tasks with labels
            for task_idx in range(batch_targets.size(1)):
                task_mask = ~torch.isnan(batch_targets[:, task_idx])
                if task_mask.any():
                    cur_true = batch_targets[:, task_idx][task_mask]
                    cur_pred = outputs[:, task_idx][task_mask]

                    w = float(self.task_weights[task_idx].item()) if self.task_weights is not None else 1.0
                    if w <= 0:
                        continue
                    batch_loss = batch_loss + w * self.criterion(cur_pred, cur_true)
                    denom += w

                    if len(task_true) <= task_idx:
                        task_true.append([])
                        task_pred.append([])

                    task_true[task_idx].append(cur_true.detach().cpu())
                    task_pred[task_idx].append(cur_pred.detach().cpu())

            if denom > 0:
                batch_loss = batch_loss / denom
            total_loss += batch_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Aggregate metrics per task (in original label space)
        all_metrics: Dict[str, Dict[str, Any]] = {}
        for task_idx in range(len(task_true)):
            true_values = torch.cat(task_true[task_idx]).numpy()
            pred_values = torch.cat(task_pred[task_idx]).numpy()

            # Denormalize to original pEC50 space if label scaling was applied
            if self.label_mean is not None and self.label_std is not None:
                mu = self.label_mean[task_idx]
                sigma = self.label_std[task_idx]
                true_orig = true_values * sigma + mu
                pred_orig = pred_values * sigma + mu
            else:
                true_orig = true_values
                pred_orig = pred_values
            
            x = pd.DataFrame(columns=['true', 'predict'])
    
            metrics = calculate_metrics_with_stats(true_orig, pred_orig)
            # Name metrics per task
            task_name = self.config.activity_columns[task_idx]
            all_metrics[task_name] = metrics

        return avg_loss, all_metrics

    # -------------------------
    # Main training loop
    # -------------------------

    def _log_split_metrics(self, split: str, epoch: int, loss: float, metrics: Dict[str, Dict[str, Any]]) -> None:
        """Print loss and per-task metrics for a dataset split."""
        print(f"seed={self.config.seed}")
        print(f"[Epoch {epoch}] {split.capitalize()} Eval Loss: {loss:.4f}")
        print(pd.DataFrame(metrics).T)

    def fit(self):
        self.load_data()
        self.model = MultiTaskMLP(self.input_dim, self.output_dim, self.config.hidden, self.config.dropout).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode="min",factor=0.5,patience=5)
        
        # Initialize TensorBoard logging
        log_dir = Path(self.config.model_dir) / "runs"
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, self.config.num_epochs+1):
            train_loss = self.train_one_epoch(epoch)
            # Evaluate current model on train / val
            train_eval_loss, train_metrics = self.evaluate(self.train_loader)
            val_loss, val_metrics = self.evaluate(self.val_loader)

            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]
            # print(f"[Epoch {epoch}] Current LR: {current_lr:.2e}")

            # TensorBoard logging
            if self.writer is not None:
                # Loss
                self.writer.add_scalar("loss/train_epoch", train_loss, epoch)
                self.writer.add_scalar("loss/train_eval", train_eval_loss, epoch)
                self.writer.add_scalar("loss/val", val_loss, epoch)
                self.writer.add_scalar("lr", current_lr, epoch)

                # Per-task metrics (train / val)
                for split_name, metrics_dict in [
                    ("train", train_metrics),
                    ("val", val_metrics),
                ]:
                    for task_name, m in metrics_dict.items():
                        base = f"{split_name}/{task_name}"
                        self.writer.add_scalar(f"{base}/mse", m["mse"], epoch)
                        self.writer.add_scalar(f"{base}/mae", m["mae"], epoch)
                        self.writer.add_scalar(f"{base}/r2", m["r2"], epoch)
                        self.writer.add_scalar(f"{base}/pearsonr", m["pearsonr"], epoch)
                        self.writer.add_scalar(f"{base}/num_samples", m["num_samples"], epoch)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_epoch = epoch
                self.best_val_loss = val_loss
                model_path = Path(self.config.model_dir) / "model.pth"
                torch.save(self.model.state_dict(), model_path)
                # print(f"[Epoch {epoch}] New best model saved to {model_path}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Simple early stopping: stop if no improvement on val for early_stop_patience epochs
            if self.config.early_stop_patience > 0 and patience_counter >= self.config.early_stop_patience:
                print(f"Validation set has no improvement for {self.config.early_stop_patience} epochs, "
                    f"stopping at epoch {epoch}.")
                break

        # After training, evaluate best model on test set
        best_model_path = model_path
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        self.model.to(self.device)
        test_loss, test_metrics = self.evaluate(self.test_loader)
        print("=" * 60)
        self._log_split_metrics("test", self.best_epoch, test_loss, test_metrics)
        print("Training completed")
  

        # After training, save all configurations
        model_config = {
            'input_dim': self.input_dim, 
            'output_dim': self.output_dim,
            'hidden': self.config.hidden,
            'dropout': self.config.dropout,
            'feature_type': self.config.feature_type,
        }
        joblib.dump(model_config, Path(self.config.model_dir) / 'model_structure.pkl')
        self.save_model_AD()

        if self.writer is not None:
            # Write final test results to TensorBoard
            self.writer.add_scalar("loss/test_final", test_loss, self.best_epoch)
            for task_name, m in test_metrics.items():
                base = f"test_final/{task_name}"
                self.writer.add_scalar(f"{base}/mse", m["mse"], self.best_epoch)
                self.writer.add_scalar(f"{base}/mae", m["mae"], self.best_epoch)
                self.writer.add_scalar(f"{base}/r2", m["r2"], self.best_epoch)
                self.writer.add_scalar(f"{base}/pearsonr", m["pearsonr"], self.best_epoch)
                self.writer.add_scalar(f"{base}/num_samples", m["num_samples"], self.best_epoch)
            self.writer.close()


    def save_model_AD(self):
        """
        Compute and save AD statistics.
        Introduce PCA to eliminate strong collinearity in high-dimensional features (ESM+RDKit+ncAA),
        and use a rigorous centered Leverage calculation scheme.
        """
        
        # 1. Extract original high-dimensional matrix
        X_scaled = self.train_AD  
        n_train = X_scaled.shape[0]
        
        
        # 3. Run PCA to eliminate collinearity and retain 90% of cumulative variance
        pca = PCA(n_components=0.90, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        p_reduced = X_pca.shape[1]  # Number of principal components retained after dimensionality reduction
        
        # 4. Calculate standard critical threshold h* for Williams plot (includes intercept degree of freedom p_reduced + 1)
        h_star = (3 * (p_reduced + 1)) / n_train
        
        print("\n" + "="*40)
        print("Applicability Domain (AD) Initialization")
        print(f"-> Original feature dimensions: {X_scaled.shape[1]}")
        print(f"-> PCA reduced dimensions: {p_reduced} (variance explained > 90%)")
        print(f"-> Training set size (n): {n_train}")
        print(f"-> Williams plot critical threshold h* = {h_star:.4f}")
        print("="*40)
        
        # 5. Compute inverse matrix in orthogonal PCA space (stable; no ridge regularization needed)
        xtx_inv = np.linalg.inv(X_pca.T @ X_pca)

        # 6. Save all key statistics and PCA model
        AD_info = {
            'pca_model': pca,
            'xtx_inv': xtx_inv,
            'h_star': h_star,
            'n_train': n_train  # Training set size for intercept term at prediction time
        }
        
        joblib.dump(AD_info, Path(self.config.model_dir) / "model_AD.pkl")

    def predict(self, new_fasta_list: List[List[str]], new_frag_list: List[List[str]], new_main_smiles_list: List[str]):
            """
            Predict activity for new peptide sequences and apply Williams AD assessment
            in the precomputed PCA space.
            """
            # 1. Path setup
            model_path = self.config.model_dir / "model.pth"
            scaler_path = Path(self.config.model_dir) / "label_scaler.npz"

            # 2. Extract features for new molecules (getEmbedding applies train feature_scaler)
            print("\nComputing feature representations for new data...")
            dataset = getEmbedding(new_fasta_list, new_frag_list, new_main_smiles_list, self.config.feature_type,
                                run_type='predict', save_path=str(self.config.model_dir))
            data_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
            predict_X = dataset.tensors[0].detach().cpu().numpy()

            input_dim = dataset.tensors[0].shape[1]
            output_dim = len(self.config.activity_columns) 

            # 3. Load MLP multi-task model and run forward pass
            model = MultiTaskMLP(input_dim, output_dim, self.config.hidden, self.config.dropout).to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            
            label_mean, label_std = None, None
            if scaler_path.exists():
                arr = np.load(scaler_path)
                label_mean = arr["mean"]
                label_std = arr["std"]
                
            all_outputs = []
            with torch.no_grad():
                for batch_embeddings, _, _ in data_loader:
                    batch_embeddings = batch_embeddings.to(self.device)
                    outputs = model(batch_embeddings)
                    all_outputs.append(outputs.cpu())
            predictions = torch.cat(all_outputs, dim=0)

            preds_np = predictions.numpy()
            # Denormalize: map from Z-score space back to pEC50 / activity values
            if label_mean is not None and label_std is not None:
                preds_np = preds_np * label_std + label_mean
            print("Multi-target peptide activity prediction complete.")

            # 4. Applicability domain (AD) assessment
            ad_path = Path(self.config.model_dir) / "model_AD.pkl"
            if not ad_path.exists():
                raise FileNotFoundError("Pretrained AD model model_AD.pkl not found; run save_model_AD() first.")
                
            ad_info = joblib.load(ad_path)
            
            pca_model = ad_info['pca_model']
            xtx_inv = ad_info['xtx_inv']
            h_star = ad_info['h_star']
            n_train = ad_info['n_train']
            
            
            # 4.2 Project new molecule features into training PCA space
            predict_X_pca = pca_model.transform(predict_X)
            
            # 4.3 Compute leverage with explicit 1/n_train intercept correction
            mahalanobis_part = np.sum((predict_X_pca @ xtx_inv) * predict_X_pca, axis=1)
            leverages = (1.0 / n_train) + mahalanobis_part
            
            # 4.4 Classify in-domain vs. out-of-domain by critical threshold
            is_inside_AD = leverages <= h_star

            print(f"AD assessment complete. In-domain samples: {np.sum(is_inside_AD)} / {len(leverages)}")

            results = {
                "predictions": preds_np,         # Shape: [N, 3] (denormalized multi-target predictions)
                "leverages": leverages,          # Shape: [N] (leverage with intercept correction)
                "is_inside_AD": is_inside_AD     # Shape: [N] (True=in domain, False=structural outlier)
            }

            return results