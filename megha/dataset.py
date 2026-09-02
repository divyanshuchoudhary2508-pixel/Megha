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
            
        # Saare text sentences ko encode karke ek long sequence banayenge
        all_tokens = []
        for item in raw_data:
            text = item["text"]
            tokens = self.tokenizer.encode(text)
            all_tokens.extend(tokens)
            
        # Agar data chhota hai (jaise abhi 5 sentences hain), 
        # toh max_seq_len ko temporary chhota kar dete hain warning se bachne ke liye
        if len(all_tokens) <= self.config.max_seq_len:
            print(f"Warning: Data size ({len(all_tokens)}) is smaller than max_seq_len ({self.config.max_seq_len}). Padding with 0s.")
            pad_len = self.config.max_seq_len - len(all_tokens) + 2
            all_tokens.extend([0] * pad_len)
            
        self.data = torch.tensor(all_tokens, dtype=torch.long)
        
    def __len__(self):
        return len(self.data) - self.config.max_seq_len - 1
        
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.config.max_seq_len]
        y = self.data[idx + 1 : idx + self.config.max_seq_len + 1]
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
