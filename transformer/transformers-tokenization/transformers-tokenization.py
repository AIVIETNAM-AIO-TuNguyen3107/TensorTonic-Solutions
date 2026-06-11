import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # Add special tokens
        self.word_to_id = {
            self.pad_token:0,
            self.unk_token:1,
            self.bos_token:2,
            self.eos_token:3,
        }
        # Build unique set of words
        tokens_set = set()
        for text in texts:
            tokens = text.lower().split()
            for token in tokens:
                tokens_set.add(token)
        # Sorted set of unique words
        tokens_set = sorted(tokens_set)

        # Build vocab
        self.word_to_id.update({
            word: id + 4
            for id, word in enumerate(tokens_set)
        })
        self.id_to_word = {
            id:word
            for word, id in self.word_to_id.items()
        }
        self.vocab_size = len(self.word_to_id.keys())

    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        tokens = text.lower().split()
        return [
            self.word_to_id[word] if word in self.word_to_id 
            else self.word_to_id[self.unk_token] # map word to id UNK if unknown
            for word in tokens
        ]
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        return " ".join([
            self.id_to_word[id] if id in self.id_to_word
            else self.id_to_word[self.word_to_id[self.unk_token]]
            for id in ids
        ])
