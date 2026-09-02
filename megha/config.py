from dataclasses import dataclass

@dataclass
class MeghaConfig:
    vocab_size: int = 10000
    d_model: int = 128      # Small for local testing (1-2M params)
    n_heads: int = 4
    n_layers: int = 2
    max_seq_len: int = 256
    dropout: float = 0.1
    batch_size: int = 4
    learning_rate: float = 1e-4
    epochs: int = 1
