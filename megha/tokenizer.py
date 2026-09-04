import os
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from .config import MeghaConfig

class MeghaTokenizer:
    def __init__(self, config: MeghaConfig):
        self.config = config
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.trainer = BpeTrainer(
            vocab_size=config.vocab_size, 
            special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
        )
        
    def train_from_iterator(self, iterator):
        self.tokenizer.train_from_iterator(iterator, self.trainer)
        
    def save(self, path):
        self.tokenizer.save(path)
        
    def load(self, path):
        self.tokenizer = Tokenizer.from_file(path)
        
    def encode(self, text):
        return self.tokenizer.encode(text).ids
        
    def decode(self, ids):
        return self.tokenizer.decode(ids)

if __name__ == "__main__":
    # Script to train the tokenizer on generated curriculum data
    print("Training Tokenizer...")
    config = MeghaConfig()
    megha_tok = MeghaTokenizer(config)
    
    # 1. Extract text into an iterator from all available curriculum files
    def text_iterator():
        import glob
        files = glob.glob("data/level_*_curriculum.json")
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    dataset = json.load(f)
                    for item in dataset:
                        if isinstance(item, dict) and "text" in item and item["text"]:
                            yield item["text"]
            except Exception:
                pass
            
    # 2. Train
    megha_tok.train_from_iterator(text_iterator())
    
    # 4. Save
    os.makedirs("data", exist_ok=True)
    megha_tok.save("data/tokenizer.json")
    print(f"Tokenizer trained and saved to data/tokenizer.json with vocab size: {megha_tok.tokenizer.get_vocab_size()}")
    
    # Quick Test
    test_text = "The computer is on."
    encoded = megha_tok.encode(test_text)
    print(f"\nTest string: '{test_text}'")
    print(f"Encoded IDs: {encoded}")
    print(f"Decoded: '{megha_tok.decode(encoded)}'")
