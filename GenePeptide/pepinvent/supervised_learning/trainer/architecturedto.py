from typing import Optional

from pydantic import BaseModel


class ArchitectureConfig(BaseModel):
    #Setup
    name: str
    training_data_path: str
    validation_data_path: str
    save_directory: str
    batch_size: int = 16
    num_epoch: int = 20
    starting_epoch: int = 1
    padding_value: int = 0
    max_sequence_length: int = 500
    #Model architecture
    N: int = 6
    H: int = 8
    d_model: int = 256
    d_ff: int = 2048
    #Regularization
    dropout: float = 0.1
    label_smoothing : float = 0.0
    #Optimization
    factor: float = 1.0
    warmup_steps: int = 4000
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-9
    lr: float = 5.0
    eval_batch_size: int = 10
    bptt: int = 35
    embedding_size: int = 200  # embedding dimension, emsize
    dimension_hid: int = 200  # dimension of the feedforward network model in nn.TransformerEncoder, d_hid
    n_layers: int = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    n_head: int = 2  # number of heads in nn.MultiheadAttention
    best_val_loss: float = float('inf')
    shuffle_each_epoch: bool = True
    drop_last_batch: bool = True
    use_cuda: bool = True
    run_type: str = 'training'

    #starting_epoch==1 && pretrained_model_path!=None：直接 Mol2MolModel.load_from_file(pretrained_model_path)
    # Fine-tuning / warm start
    # If provided (and starting_epoch == 1), the trainer will load the Mol2MolModel from this path
    # and use the model's stored vocabulary for tokenization.
    pretrained_model_path: Optional[str] = None
    # If True and pretrained_model_path is provided, scan the new dataset and extend the pretrained vocabulary
    # with any missing tokens, then resize embeddings/generator safely.
    # 扫描新训练/验证 CSV 的 Source_Mol/Target_Mol，找出预训练 vocab 里没有的新 token（按排序保证确定性）
    extend_vocabulary: bool = False
    # Optional optimizer checkpoint to warm-start optimization when fine-tuning from pretrained_model_path.
    pretrained_optimizer_path: Optional[str] = None
    # If True, ignore pretrained_optimizer_path (or any saved optimizer) and re-initialize optimizer.
    # reset_optimizer：true 就重新初始化 NoamOpt；否则如果给了 pretrained_optimizer_path 就加载，否则退回重新初始化。
    reset_optimizer: bool = False

    # --- DDP sanity check (optional) ---
    # If enabled, the trainer will periodically compare a small slice of a model parameter across ranks.
    # This helps detect incorrect DDP usage where gradients/weights diverge across ranks.
    ddp_param_check_enabled: bool = False
    ddp_param_check_interval_steps: int = 2000
    ddp_param_check_numel: int = 2048
    ddp_param_check_rtol: float = 1e-5
    ddp_param_check_atol: float = 1e-7

    # --- Validation similarity (RDKit) ---
    # NOTE: RDKit fingerprinting can be CPU-heavy. In DDP, if one rank is slower,
    # others can block on all_reduce and hit NCCL watchdog timeout.
    # Use these knobs to cap work and keep validation stable.
    validation_similarity_enabled: bool = True
    validation_similarity_max_pairs_per_rank: int = 200
