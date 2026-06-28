import os
import random
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from rdkit import Chem
from embedding import getEmbedding

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_split import stratify_split_indices, random_split_indices
import model_base_pooling as model_base
# import model_base
from pathlib import Path


@dataclass
class TrainConfigNew:
    """Training configuration"""
    # Data and paths
    run_type: str  # 'train' or 'predict'
    csv_path: str
    model_dir: str
    peptide_seq_name: str
    peptide_smis_name: str
    feature_type: List[str]
    activity_columns: List[str]
    predict_columns: List[str] = None  # Output column names when predicting
    split_ratio: float = 0.1  # Training/validation/test split ratio
    split_method: str = "random"  # 'random' or 'stratify'

    # Training hyperparameters
    seed: int = 42
    batch_size: int = 64
    num_epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-5
    early_stop_patience: int = 10 


    # Model structure
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    dropout: float = 0.1


def fix_seed_new(seed: int = 42) -> None:
    """Fix random seed (safer for non-CUDA cases)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_metrics_with_stats(true: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    return {
        "rmse":np.round(np.sqrt(mean_squared_error(true, pred)), 4),
        "mae": mean_absolute_error(true, pred),
        "r2": r2_score(true, pred),
        "pearsonr":np.corrcoef(true, pred)[0, 1],
        "num_samples": len(true),
    }


class MultiTaskTransformerTrainer:
    def __init__(self, config: TrainConfigNew):
        self.config = config
        fix_seed_new(config.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loader: DataLoader = None
        self.val_loader: DataLoader = None
        self.test_loader: DataLoader = None
        self.predict_loader: DataLoader = None
        self.model: nn.Module = None
        self.optimizer: torch.optim.Optimizer = None
        self.scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau = None
        self.criterion: nn.Module = nn.MSELoss()

        self.best_val_loss = float("inf")
        self.best_epoch = 1
        self.writer: SummaryWriter | None = None
        # Standardization parameters for each task label (calculated on training set)
        self.label_mean: np.ndarray | None = None
        self.label_std: np.ndarray | None = None

    # -------------------------
    # Build data / model / optimizer
    # -------------------------

    def load_data(self) -> None:
        # Read specified columns
        self.df = pd.read_csv(self.config.csv_path)
        fasta_list = [eval(fasta) for fasta in self.df[self.config.peptide_seq_name].to_list()]
        frag_list = [eval(frag) for frag in self.df[self.config.peptide_smis_name].to_list()]
        self.config.model_dir.mkdir(parents=True, exist_ok=True)

        activity_array = self.df[self.config.activity_columns].values

        # Split before calculating embedding to avoid standardization leakage
        train_idx, val_idx, test_idx = stratify_split_indices(
                activity_array,
                val_ratio=self.config.split_ratio,
                test_ratio=self.config.split_ratio,
                seed=self.config.seed,
                n_bins=3,
            )

        print(f'train_number:{len(train_idx)}, val_number:{len(val_idx)}, test_number:{len(test_idx)}')
        self.df['TrTe'] = 'Train'
        self.df.loc[val_idx, 'TrTe'] = 'Validation'
        self.df.loc[test_idx, 'TrTe'] = 'Test'
        self.df.to_csv(Path(self.config.model_dir) / Path(self.config.csv_path.replace('.csv', '_split.csv')).name, index=None)


        # Split fasta / frag / activity by indices
        fasta_train, frag_train = [fasta_list[i] for i in train_idx], [frag_list[i] for i in train_idx]
        fasta_val, frag_val = [fasta_list[i] for i in val_idx], [frag_list[i] for i in val_idx]
        fasta_test, frag_test = [fasta_list[i] for i in test_idx], [frag_list[i] for i in test_idx]

        activity_train = activity_array[train_idx] if activity_array is not None else None
        activity_val = activity_array[val_idx] if activity_array is not None else None
        activity_test = activity_array[test_idx] if activity_array is not None else None

        # Standardize each task label by training set (z-score)
        if activity_train is not None:
            # [n_train, n_tasks]
            mean = np.nanmean(activity_train, axis=0)
            std = np.nanstd(activity_train, axis=0)
            # Handle variance of 0 or NaN
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

            scaler_path = Path(self.config.model_dir) / "label_scaler.npz"
            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(scaler_path, mean=mean, std=std)

        # Calculate embedding on training set and fit scaler, save to model_dir
        train_dataset = getEmbedding(fasta_list=fasta_train, frag_list=frag_train, run_type="train",\
                                     feature_type=self.config.feature_type,\
                                    activity_array=activity_train, save_path=self.config.model_dir
        )

        # Use same scaler for validation and test sets
        val_dataset = getEmbedding(fasta_list=fasta_val, frag_list=frag_val, run_type="predict",\
                                    feature_type=self.config.feature_type,\
                                    activity_array=activity_val, save_path=self.config.model_dir
        )
        test_dataset = getEmbedding(fasta_list=fasta_test, frag_list=frag_test, run_type="predict",\
                                    feature_type=self.config.feature_type,\
                                    activity_array=activity_test, save_path=self.config.model_dir
        )

        self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)

        self.val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

        self.test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        # Get input and output dimensions
        emb, mask, target = train_dataset[0]
        self.input_dim = emb.shape[1] # # [max_len, d_model]
        self.output_dim = len(self.config.activity_columns) #共3个任务
        print(f'Input dimension:{self.input_dim}, output dimension:{self.output_dim}')



    def _build_model(self) -> None:
        self.model = model_base.TransformerRegressor(
            input_dim=self.input_dim,
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            num_layers=self.config.num_layers,
            output_dim=self.output_dim,
            dropout=self.config.dropout,
            max_len=45
        ).to(self.device)

    def _build_optimization(self) -> None:
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

    # -------------------------
    # Single epoch training / validation
    # -------------------------

    def train_one_epoch(self, epoch: int) -> float:
        """Single epoch training for multi-task regression"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_embeddings, batch_masks, batch_targets in self.train_loader:
            batch_embeddings = batch_embeddings.to(self.device)
            batch_masks = batch_masks.to(self.device)
            batch_targets = batch_targets.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(batch_embeddings, batch_masks)  # [batch, num_tasks]

            loss = 0.0
            num_active_tasks = 0

            for task_idx in range(batch_targets.size(1)):
                task_mask = ~torch.isnan(batch_targets[:, task_idx])
                if task_mask.any():
                    loss = loss + self.criterion(
                        outputs[:, task_idx][task_mask],
                        batch_targets[:, task_idx][task_mask],
                    )
                    num_active_tasks += 1

            loss = loss / num_active_tasks
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"[Epoch {epoch}] Train Opt Loss: {avg_loss:.4f}")
        return avg_loss
        

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, Dict[str, Any]]]:
        """Evaluate: return average loss and metrics for each task"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        task_true = []
        task_pred = []

        for batch_embeddings, batch_masks, batch_targets in dataloader:
            batch_embeddings = batch_embeddings.to(self.device)
            batch_masks = batch_masks.to(self.device)
            batch_targets = batch_targets.to(self.device)

            outputs = self.model(batch_embeddings, batch_masks)

            batch_loss = 0.0
            num_active_tasks = 0
            # Only backpropagate for tasks with labels
            for task_idx in range(batch_targets.size(1)):
                task_mask = ~torch.isnan(batch_targets[:, task_idx])
                if task_mask.any():
                    cur_true = batch_targets[:, task_idx][task_mask]
                    cur_pred = outputs[:, task_idx][task_mask]

                    batch_loss = batch_loss + self.criterion(cur_pred, cur_true)
                    num_active_tasks += 1

                    if len(task_true) <= task_idx:
                        task_true.append([])
                        task_pred.append([])

                    task_true[task_idx].append(cur_true.detach().cpu())
                    task_pred[task_idx].append(cur_pred.detach().cpu())

            batch_loss = batch_loss / num_active_tasks
            total_loss += batch_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Summarize metrics by task (calculated on original label space)
        all_metrics: Dict[str, Dict[str, Any]] = {}
        for task_idx in range(len(task_true)):
            true_values = torch.cat(task_true[task_idx]).numpy()
            pred_values = torch.cat(task_pred[task_idx]).numpy()

            if self.label_mean is not None and self.label_std is not None:
                mu = self.label_mean[task_idx]
                sigma = self.label_std[task_idx]
                true_orig = true_values * sigma + mu
                pred_orig = pred_values * sigma + mu
            else:
                true_orig = true_values
                pred_orig = pred_values

            metrics = calculate_metrics_with_stats(true_orig, pred_orig)
          
            task_name = self.config.activity_columns[task_idx]
            all_metrics[task_name] = metrics

        return avg_loss, all_metrics

    # -------------------------
    # Training main loop
    # -------------------------

    def _log_split_metrics(self, split: str, epoch: int, loss: float, metrics: Dict[str, Dict[str, Any]]) -> None:
        """Print loss and metrics for a specific dataset (split)"""
        print(f"seed={self.config.seed}")
        print(f"[Epoch {epoch}] {split.capitalize()} Eval Loss: {loss:.4f}")
        for name, m in metrics.items():
            print(
                f"  [{split.capitalize()}] {name}: "
                f"RMSE={m['rmse']:.4f}, "
                f"MAE={m['mae']:.4f}, "
                f"R2={m['r2']:.4f}, "
                f"Pearson={m['pearsonr']:.4f}, "
                f"Num={m['num_samples']}"
            )

    def fit(self) -> None:
        """Complete training process"""
        self.load_data()
        self._build_model()

        print("\n" + "=" * 60)
        print('Execute model training......')
        print(type(self.config.activity_columns), self.config.activity_columns)
        self._build_optimization()

        # Initialize TensorBoard log
        log_dir = Path(self.config.model_dir) / "runs"
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))
        
        # Early stopping counter
        no_improve_epochs = 0

        for epoch in range(1, self.config.num_epochs + 1):
            train_loss = self.train_one_epoch(epoch)

            # Evaluate current model on train / val
            train_eval_loss, train_metrics = self.evaluate(self.train_loader)
            val_loss, val_metrics = self.evaluate(self.val_loader)

            self._log_split_metrics("train", epoch, train_eval_loss, train_metrics)
            self._log_split_metrics("val", epoch, val_loss, val_metrics)

            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"[Epoch {epoch}] Current LR: {current_lr:.2e}")


            if self.writer is not None:
                self.writer.add_scalar("loss/train_epoch", train_loss, epoch)
                self.writer.add_scalar("loss/train_eval", train_eval_loss, epoch)
                self.writer.add_scalar("loss/val", val_loss, epoch)
                self.writer.add_scalar("lr", current_lr, epoch)

                for split_name, metrics_dict in [
                    ("train", train_metrics),
                    ("val", val_metrics),
                ]:
                    for task_name, m in metrics_dict.items():
                        base = f"{split_name}/{task_name}"
                        self.writer.add_scalar(f"{base}/rmse", m["rmse"], epoch)
                        self.writer.add_scalar(f"{base}/mae", m["mae"], epoch)
                        self.writer.add_scalar(f"{base}/r2", m["r2"], epoch)
                        self.writer.add_scalar(f"{base}/pearsonr", m["pearsonr"], epoch)
                        self.writer.add_scalar(f"{base}/num_samples", m["num_samples"], epoch)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_epoch = epoch
                self.best_val_loss = val_loss
                model_path = Path(self.config.model_dir) / "model.pth"
                os.makedirs(self.config.model_dir, exist_ok=True)
                torch.save(self.model.state_dict(), model_path)
                print(f"[Epoch {epoch}] New best model saved to {model_path}")
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            # Simple early stopping: stop if no improvement on validation set for early_stop_patience consecutive epochs
            if self.config.early_stop_patience > 0 and no_improve_epochs >= self.config.early_stop_patience:
                print(f"验证集在连续 {self.config.early_stop_patience} 个 epoch 上无提升，"
                    f"在第 {epoch} 个 epoch 提前停止训练。")
                break

        # After training, evaluate using the best model on validation set on test set
        best_model_path = Path(self.config.model_dir) / "model.pth"
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        self.model.to(self.device)
        test_loss, test_metrics = self.evaluate(self.test_loader)
        print("=" * 60)
        self._log_split_metrics("test", self.best_epoch, test_loss, test_metrics)
        print("Training completed")
        if self.writer is not None:
            self.writer.add_scalar("loss/test_final", test_loss, self.best_epoch)
            for task_name, m in test_metrics.items():
                base = f"test_final/{task_name}"
                self.writer.add_scalar(f"{base}/rmse", m["rmse"], self.best_epoch)
                self.writer.add_scalar(f"{base}/mae", m["mae"], self.best_epoch)
                self.writer.add_scalar(f"{base}/r2", m["r2"], self.best_epoch)
                self.writer.add_scalar(f"{base}/pearsonr", m["pearsonr"], self.best_epoch)
                self.writer.add_scalar(f"{base}/num_samples", m["num_samples"], self.best_epoch)
            self.writer.close()

#######Y
    def predict(self, fasta_list: List[List[str]], frag_list: List[List[str]]):
        print("\n" + "=" * 60)
        print('Execute model prediction......')
        

        all_dataset = getEmbedding(fasta_list=fasta_list, frag_list=frag_list, run_type="predict",
            feature_type=self.config.feature_type, activity_array=None, save_path=self.config.model_dir
            )
        predict_loader = DataLoader(all_dataset, batch_size=self.config.batch_size, shuffle=False
                                    )
        emb, mask = all_dataset[0]
        input_dim = emb.shape[1] # # [max_len, d_model]
        output_dim = len(self.config.activity_columns)
        model_path = Path(self.config.model_dir) / "model.pth"
        model = model_base.TransformerRegressor(input_dim=input_dim, d_model=self.config.d_model, nhead=self.config.nhead,\
                                                num_layers=self.config.num_layers, output_dim=output_dim,\
                                                dropout=self.config.dropout, max_len=45).to(self.device)
        model.load_state_dict(torch.load(model_path))
        model.eval()



        scaler_path = Path(self.config.model_dir) / "label_scaler.npz"
        label_mean = None
        label_std = None
        if scaler_path.exists():
            arr = np.load(scaler_path)
            label_mean = arr["mean"]
            label_std = arr["std"]
        all_outputs = []

        with torch.no_grad():
            for batch_embeddings, batch_masks in predict_loader:
                batch_embeddings = batch_embeddings.to(self.device)
                batch_masks = batch_masks.to(self.device)
                outputs = model(batch_embeddings, batch_masks)
                all_outputs.append(outputs.cpu())
        predictions = torch.cat(all_outputs, dim=0)
        preds_np = predictions.numpy()
       
        if label_mean is not None and label_std is not None:
            preds_np = preds_np * label_std + label_mean
        return preds_np