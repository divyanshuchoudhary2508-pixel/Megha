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
        
        vocab = tokenizer.tokenizer.get_vocab()
        eos_id = vocab.get("<|endoftext|>", 0)
        
        all_x_tokens = []
        all_y_tokens = []
        
        for text in all_texts:
            if not text or "Q:" not in text or "A:" not in text:
                continue
            parts = text.split("A:", 1)
            prompt_str = parts[0] + "A:"
            answer_str = parts[1]
            
            prompt_ids = self.tokenizer.encode(prompt_str)
            answer_ids = self.tokenizer.encode(answer_str)
            if not answer_ids:
                continue
            answer_ids.append(eos_id)
            
            # Input: prompt + answer
            x_seq = prompt_ids + answer_ids
            # Target: -100 for prompt tokens (no loss gradient), answer_ids for answer tokens
            y_seq = [-100] * len(prompt_ids) + answer_ids
            
            all_x_tokens.extend(x_seq)
            all_y_tokens.extend(y_seq)
            
        # Pad with EOS / -100 if sequence is short
        min_len = self.config.max_seq_len + 1
        while len(all_x_tokens) < min_len:
            all_x_tokens.extend([eos_id] * 20)
            all_y_tokens.extend([-100] * 20)
            
        self.x_data = torch.tensor(all_x_tokens, dtype=torch.long)
        self.y_data = torch.tensor(all_y_tokens, dtype=torch.long)
        
        self.stride = max(1, self.config.max_seq_len // 2)
        
    def __len__(self):
        return max(1, (len(self.x_data) - self.config.max_seq_len - 1) // self.stride)
        
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        x = self.x_data[start_idx : start_idx + self.config.max_seq_len]
        # Shift target by 1 token for standard next-token prediction
        y = self.y_data[start_idx + 1 : start_idx + self.config.max_seq_len + 1]
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
