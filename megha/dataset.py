import json
import glob
import torch
import random
from torch.utils.data import Dataset, DataLoader
from .tokenizer import MeghaTokenizer
from .config import MeghaConfig
import os

class MeghaDataset(Dataset):
    def __init__(self, all_texts: list, tokenizer: MeghaTokenizer, config: MeghaConfig):
        self.config = config
        self.tokenizer = tokenizer
        
        # EOS separator between each Q&A example
        eos_tokens = self.tokenizer.encode("<|endoftext|>")
        if not eos_tokens:
            eos_tokens = [0]
            
        all_tokens = []
        for text in all_texts:
            if not text:
                continue
            tokens = self.tokenizer.encode(text)
            if tokens:
                all_tokens.extend(tokens)
                all_tokens.extend(eos_tokens)
            
        # Pad if still too small
        while len(all_tokens) <= self.config.max_seq_len + 1:
            all_tokens.extend(eos_tokens * 10)
            
        self.data = torch.tensor(all_tokens, dtype=torch.long)
        # 50% stride: good balance between coverage and overfitting prevention
        self.stride = max(1, self.config.max_seq_len // 2)
        
    def __len__(self):
        return max(1, (len(self.data) - self.config.max_seq_len - 1) // self.stride)
        
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        x = self.data[start_idx : start_idx + self.config.max_seq_len]
        y = self.data[start_idx + 1 : start_idx + self.config.max_seq_len + 1]
        return x, y


def load_texts_from_file(data_path: str) -> list:
    """Load all Q&A texts from a single curriculum JSON file."""
    if not os.path.exists(data_path):
        return []
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        texts = []
        for item in raw_data:
            text = item.get("text", "")
            if text and len(text.strip()) > 5:
                texts.append(text.strip())
        return texts
    except Exception as e:
        print(f"Warning: Could not load {data_path}: {e}")
        return []


def get_combined_dataloader(tokenizer_path: str, config: MeghaConfig, data_dir: str = "data"):
    """
    MIXED TRAINING: Load ALL levels' data, shuffle together, train ONE model.
    This prevents Catastrophic Forgetting.
    """
    tokenizer = MeghaTokenizer(config)
    if os.path.exists(tokenizer_path):
        tokenizer.load(tokenizer_path)
    else:
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Run tokenizer.py first.")
    
    # Load all curriculum files
    all_texts = []
    files = sorted(glob.glob(f"{data_dir}/level_*_curriculum.json"))
    
    for fpath in files:
        texts = load_texts_from_file(fpath)
        all_texts.extend(texts)
        print(f"  Loaded {len(texts)} examples from {os.path.basename(fpath)}")
    
    if not all_texts:
        raise ValueError("No training data found! Run data_gen.py first.")
    
    # SHUFFLE: mix all levels together so model learns all topics uniformly
    random.shuffle(all_texts)
    print(f"\nTotal training examples (all levels combined): {len(all_texts)}")
    
    dataset = MeghaDataset(all_texts, tokenizer, config)
    print(f"Total dataset chunks: {len(dataset)}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False
    )
    return dataloader, tokenizer


def get_dataloader(data_path: str, tokenizer_path: str, config: MeghaConfig):
    """Single-level loader (kept for backward compatibility)."""
    tokenizer = MeghaTokenizer(config)
    if os.path.exists(tokenizer_path):
        tokenizer.load(tokenizer_path)
    else:
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Run tokenizer.py first.")
    
    texts = load_texts_from_file(data_path)
    dataset = MeghaDataset(texts, tokenizer, config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False
    )
    return dataloader, tokenizer
