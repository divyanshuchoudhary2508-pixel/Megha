from dataclasses import dataclass

@dataclass
class MeghaConfig:
    vocab_size: int = 8000    # 8k vocabulary — captures complete CloudOps words without splitting
    max_seq_len: int = 256
    d_model: int = 384        # Upgraded width
    n_layers: int = 8         # 8 Transformer layers
    n_heads: int = 6          # 6 Attention heads (384 / 6 = 64 head dim)
    dropout: float = 0.1
    batch_size: int = 8       # Batch size 8 for stable gradients on 15M model
    learning_rate: float = 3e-4 # Standard stable LR for ~15M model
    epochs: int = 15          # 15 epochs on 15M model gives optimal convergence without overfitting
