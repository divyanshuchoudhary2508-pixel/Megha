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
        pad_id = vocab.get("[PAD]", 1)
        eos_id = vocab.get("<|endoftext|>", 0)
        
        self.samples_x = []
        self.samples_y = []
        
        for text in all_texts:
            if not text:
                continue
                
            prompt_str = ""
            answer_str = ""
            
            # Format 1: ChatML format
            if "<|im_start|>user" in text and "<|im_start|>assistant" in text:
                parts = text.split("<|im_start|>assistant", 1)
                prompt_str = parts[0] + "<|im_start|>assistant\n"
                answer_str = parts[1].replace("<|im_end|>", "").strip() + "<|im_end|>"
            # Format 2: Standard Q: ... A: ... format
            elif "Q:" in text and "A:" in text:
                parts = text.split("A:", 1)
                prompt_str = "<|im_start|>user\n" + parts[0].replace("Q:", "").strip() + "<|im_end|>\n<|im_start|>assistant\n"
                answer_str = parts[1].strip() + "<|im_end|>"
            else:
                continue
                
            prompt_ids = self.tokenizer.encode(prompt_str)
            ans_ids = self.tokenizer.encode(answer_str)
            if not ans_ids:
                continue
            ans_ids.append(eos_id)
            
            x_raw = prompt_ids + ans_ids
            y_raw = [-100] * len(prompt_ids) + ans_ids
            
            # Truncate to max_seq_len if too long
            if len(x_raw) > self.config.max_seq_len:
                x_raw = x_raw[:self.config.max_seq_len]
                y_raw = y_raw[:self.config.max_seq_len]
            else:
                # Pad to max_seq_len
                pad_len = self.config.max_seq_len - len(x_raw)
                x_raw = x_raw + [pad_id] * pad_len
                y_raw = y_raw + [-100] * pad_len
                
            self.samples_x.append(torch.tensor(x_raw, dtype=torch.long))
            self.samples_y.append(torch.tensor(y_raw, dtype=torch.long))
            
        if not self.samples_x:
            # Fallback dummy sample
            dummy_x = torch.zeros(self.config.max_seq_len, dtype=torch.long)
            dummy_y = torch.full((self.config.max_seq_len,), -100, dtype=torch.long)
            self.samples_x = [dummy_x]
            self.samples_y = [dummy_y]
            
    def __len__(self):
        return len(self.samples_x)
        
    def __getitem__(self, idx):
        return self.samples_x[idx], self.samples_y[idx]


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
