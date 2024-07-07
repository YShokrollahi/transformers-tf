import unittest
from transformer.train import train_transformer

class TestTransformer(unittest.TestCase):
    def test_train_transformer(self):
        transformer, tokenizer_pt, tokenizer_en = train_transformer(EPOCHS=1)
        self.assertIsNotNone(transformer)
        self.assertIsNotNone(tokenizer_pt)
        self.assertIsNotNone(tokenizer_en)

if __name__ == "__main__":
    unittest.main()
