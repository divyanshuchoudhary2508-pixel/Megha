from dataclasses import dataclass

@dataclass
class MeghaConfig:
    vocab_size: int = 8000        # 8k ByteLevel vocabulary
    max_seq_len: int = 256        # Standard sequence length for ChatML Q&A
    d_model: int = 384            # 17.37M sweet spot parameters
    n_layers: int = 8             # 8 Transformer blocks
    n_heads: int = 6              # 6 heads (64 dim per head)
    dropout: float = 0.10         # Standard dropout
    batch_size: int = 8           # Batch size 8
    grad_accum_steps: int = 2     # Effective batch size = 16
    learning_rate: float = 1e-3   # High LR for SLM fast convergence (Karpathy / SmolLM standard)
    epochs: int = 15              # 15 epochs on 30,000 ChatML dataset (~28,000 steps)
    warmup_steps: int = 300       # 300 warmup steps
