from dataclasses import dataclass

@dataclass
class MeghaConfig:
    vocab_size: int = 8000        # 8k ByteLevel vocabulary
    max_seq_len: int = 384        # Extended context window for multi-sentence answers
    d_model: int = 512            # Upgraded width -> 32.5M parameters
    n_layers: int = 10            # 10 Transformer blocks
    n_heads: int = 8              # 8 heads (64 dim per head)
    dropout: float = 0.1
    batch_size: int = 4           # Small batch for gradient stability
    grad_accum_steps: int = 2     # Effective batch size = 8
    learning_rate: float = 3e-4
    epochs: int = 30              # 30 epochs on full dataset
    warmup_steps: int = 150       # Linear LR Warmup steps
