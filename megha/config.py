from dataclasses import dataclass

@dataclass
class MeghaConfig:
    vocab_size: int = 8000
    max_seq_len: int = 512
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    dropout: float = 0.1
    batch_size: int = 16
    learning_rate: float = 3e-4
    epochs: int = 3
