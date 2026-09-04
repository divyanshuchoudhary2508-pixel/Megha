from dataclasses import dataclass

@dataclass
class MeghaConfig:
    vocab_size: int = 4000
    max_seq_len: int = 256
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    dropout: float = 0.1
    batch_size: int = 32
    learning_rate: float = 5e-4
    epochs: int = 6
