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
    
    # 1. Load data from the data folder
    data_path = "data/level_0_curriculum.json"
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found. Run data_gen.py first.")
        exit(1)
        
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    # 2. Extract text into an iterator
    def text_iterator():
        for item in dataset:
            yield item["text"]
            
    # 3. Train
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
