import unittest
import torch
import os
import json
import tempfile
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from megha.config import MeghaConfig
from megha.model import MeghaModel
from megha.tokenizer import MeghaTokenizer
from megha.dataset import MeghaDataset, load_texts_from_file


class TestMeghaArchitecture(unittest.TestCase):
    def setUp(self):
        self.config = MeghaConfig(
            vocab_size=1000,
            max_seq_len=64,
            d_model=128,
            n_layers=2,
            n_heads=2,
            batch_size=2
        )
        self.model = MeghaModel(self.config)

    def test_weight_tying(self):
        """Verify token embedding weights and LM head weights are strictly tied (same memory pointer)."""
        self.assertTrue(
            self.model.token_emb.weight is self.model.lm_head.weight,
            "token_emb.weight and lm_head.weight MUST be tied!"
        )

    def test_forward_pass_shapes(self):
        """Verify forward pass output logits shape and loss calculation."""
        B, T = 2, 16
        x = torch.randint(0, self.config.vocab_size, (B, T))
        y = torch.randint(0, self.config.vocab_size, (B, T))
        # Set half targets to -100 for instruction loss masking test
        y[:, :8] = -100

        logits, loss = self.model(x, targets=y)
        self.assertEqual(logits.shape, (B, T, self.config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0.0)

    def test_generation(self):
        """Verify text generation produces valid token sequences."""
        x = torch.tensor([[10, 20, 30]], dtype=torch.long)
        out = self.model.generate(x, max_new_tokens=10, temperature=0.7, top_k=20)
        self.assertEqual(out.shape, (1, 13))


class TestMeghaTokenizerAndDataset(unittest.TestCase):
    def setUp(self):
        self.config = MeghaConfig(
            vocab_size=1000,
            max_seq_len=64,
            d_model=128,
            n_layers=2,
            n_heads=2,
            batch_size=2
        )
        self.tokenizer = MeghaTokenizer(self.config)
        
        # Train tokenizer on sample texts
        sample_texts = [
            "<|im_start|>user\nWhat is Linux?<|im_end|>\n<|im_start|>assistant\nLinux is an open-source operating system kernel.<|im_end|>",
            "<|im_start|>user\nWhat is EC2?<|im_end|>\n<|im_start|>assistant\nAmazon EC2 provides scalable compute capacity.<|im_end|>",
            "Q: What does Docker do?\nA: Docker packages applications into containers."
        ]
        self.tokenizer.train_from_iterator(sample_texts)

    def test_tokenizer_chatml_special_tokens(self):
        """Verify ChatML special tokens are registered correctly in vocabulary."""
        vocab = self.tokenizer.tokenizer.get_vocab()
        self.assertIn("<|im_start|>", vocab)
        self.assertIn("<|im_end|>", vocab)
        self.assertIn("<|endoftext|>", vocab)

    def test_dataset_chatml_sft_masking(self):
        """Verify dataset builds ChatML sequences with discrete sample isolation & target masking."""
        sample_texts = [
            "<|im_start|>user\nWhat is Linux?<|im_end|>\n<|im_start|>assistant\nLinux is an OS kernel.<|im_end|>",
            "Q: What is Docker?\nA: Docker containers isolate applications."
        ]
        dataset = MeghaDataset(sample_texts, self.tokenizer, self.config)
        self.assertEqual(len(dataset), 2)

        x, y = dataset[0]
        self.assertEqual(x.shape, (self.config.max_seq_len,))
        self.assertEqual(y.shape, (self.config.max_seq_len,))
        
        # Check that user prompt target tokens are masked with -100
        first_few_y = y[:5].tolist()
        self.assertIn(-100, first_few_y, "Target prompt tokens must be masked with -100")

    def test_load_texts_from_file(self):
        """Verify load_texts_from_file safely reads curriculum JSON files."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as tmp:
            json.dump([{"text": "Q: Test question?\nA: Test answer."}], tmp)
            tmp_path = tmp.name

        try:
            texts = load_texts_from_file(tmp_path)
            self.assertEqual(len(texts), 1)
            self.assertIn("Q: Test question?", texts[0])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
