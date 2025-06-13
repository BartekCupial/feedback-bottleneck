from functools import lru_cache
from typing import Dict

import numpy as np
from transformers import AutoTokenizer


class Tokenizer:
    LRU_CACHE_SIZE = 1000

    def __init__(self, max_token_length: int = 128, tokenizer_id: str = "distilroberta-base"):
        self.max_token_length = max_token_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, truncation_side="left")

    # We use caching to avoid re-tokenizing data that are already seen.
    @lru_cache(maxsize=LRU_CACHE_SIZE)
    def _tokenize(self, input_text: str):
        tokens = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_token_length,
        )
        return tokens.data

    def decode(self, input_ids: np.ndarray) -> str:
        """Decode input_ids to a string."""
        return self.tokenizer.decode(input_ids, skip_special_tokens=False)

    def __call__(self, text_str) -> Dict[str, np.ndarray]:
        tokenized = {}
        tokens = self._tokenize(text_str)
        tokenized["input_ids"] = tokens["input_ids"][0].numpy()
        tokenized["attention_mask"] = tokens["attention_mask"][0].numpy()

        return tokenized
