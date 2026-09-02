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
