import json
import torch
from torch.utils.data import Dataset, DataLoader
from .tokenizer import MeghaTokenizer
from .config import MeghaConfig
import os

class MeghaDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: MeghaTokenizer, config: MeghaConfig):
        self.config = config
        self.tokenizer = tokenizer
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} not found. Run data_gen.py first.")
            
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            
        # Saare text sentences ko encode karke ek sequence banayenge with EOS separator
        eos_tokens = self.tokenizer.encode("<|endoftext|>")
        if not eos_tokens:
            eos_tokens = [0]
            
        all_tokens = []
        for item in raw_data:
            text = item.get("text", "")
            if not text:
                continue
            tokens = self.tokenizer.encode(text)
            all_tokens.extend(tokens)
            all_tokens.extend(eos_tokens)
            
        # Agar data chhota hai, pad with EOS
        if len(all_tokens) <= self.config.max_seq_len:
            pad_len = self.config.max_seq_len - len(all_tokens) + 2
            all_tokens.extend(eos_tokens * (pad_len // len(eos_tokens) + 1))
            
        self.data = torch.tensor(all_tokens, dtype=torch.long)
        # Stride = max_seq_len // 2 (gives 2x coverage instead of 512x crazy overfitting)
        self.stride = max(1, self.config.max_seq_len // 2)
        
    def __len__(self):
        return max(1, (len(self.data) - self.config.max_seq_len - 1) // self.stride)
        
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        x = self.data[start_idx : start_idx + self.config.max_seq_len]
        y = self.data[start_idx + 1 : start_idx + self.config.max_seq_len + 1]
        return x, y

def get_dataloader(data_path: str, tokenizer_path: str, config: MeghaConfig):
    tokenizer = MeghaTokenizer(config)
    
    if os.path.exists(tokenizer_path):
        tokenizer.load(tokenizer_path)
    else:
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Run tokenizer.py first.")
        
    dataset = MeghaDataset(data_path, tokenizer, config)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        drop_last=True
    )
    return dataloader, tokenizer
