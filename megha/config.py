from dataclasses import dataclass

@dataclass
class MeghaConfig:
    vocab_size: int = 8000        # 8k ByteLevel vocabulary
    max_seq_len: int = 256
    d_model: int = 384            # 17.37M params
    n_layers: int = 8             # 8 Transformer blocks
    n_heads: int = 6              # 6 heads (64 dim per head)
    dropout: float = 0.1
    batch_size: int = 4           # Small batch for gradient stability & higher step resolution
    grad_accum_steps: int = 2     # Effective batch size = 8
    learning_rate: float = 3e-4
    epochs: int = 30              # 30 epochs on full ~3,500+ dataset for maximum mastery
    warmup_steps: int = 150       # Linear LR Warmup steps
