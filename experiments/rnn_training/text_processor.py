"""
This source code is part of a deep learning coursework and research framework.
It is provided for educational use, experimentation, and academic projects.

You are free to use, modify, and redistribute this code for personal, academic,
or research purposes.

# Author: mahdi mansouri
# GitHub: https://github.com/mehtimans
# Date: July 2026
"""
import re
import torch
from typing import Tuple, List, Dict
from typing import Counter as CounterType
from collections import Counter

def load_data(path_pos: str, path_neg: str) -> Tuple[List[str], List[int]]:
    """Load pos/neg files, return (texts, labels) with 1=pos, 0=neg."""
    texts, labels = [], []
    for path, label in [(path_pos, 1), (path_neg, 0)]:
        with open(path, encoding="latin-1") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
                    labels.append(label)
    return texts, labels

def whitespace_tokenizer(text: str) -> List[str]:
    """
    Convert text to lowercase, remove punctuation,
    and split into whitespace-separated tokens.
    """

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()

def split_dataset(
    X: List[List[str]],
    Y: List[int],
    val_split: float = 0.2,
    test_split: float = 0.0,
) -> Tuple[List[List[str]], List[int],
           List[List[str]], List[int],
           List[List[str]], List[int]]:
    """
    Split a tokenized classification dataset into training,
    validation, and test sets while preserving class proportions.
    """  
    if len(X) != len(Y):
        raise ValueError(f"X and Y must have the same length, got {len(x)} and {len(Y)}.") 

    if not 0.0 <= val_split < 1.0:
        raise ValueError("val_split must be in range [0, 1).")

    if not 0.0 <= test_split < 1.0:
        raise ValueError("test_split must be in the range [0, 1).")

    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be smaller than 1.") 

    train_indices = []
    val_indices = []
    test_indices = []

    # Split each Class independently
    for label in set(Y): 

        indices = [i for i, y in enumerate(Y)
                     if y == label] 

        permutation = torch.randperm(len(indices)).tolist()

        indices = [indices[i] for i in permutation]

        n_val = int(len(indices) * val_split)
        n_test = int(len(indices) * test_split)

        val_indices.extend(indices[:n_val])

        test_indices.extend(indices[n_val:n_val + n_test])

        train_indices.extend(indices[n_val + n_test:])

    X_train = [X[i] for i in train_indices]
    Y_train = [Y[i] for i in train_indices]

    X_val = [X[i] for i in val_indices]
    Y_val = [Y[i] for i in val_indices]

    X_test = [X[i] for i in test_indices]
    Y_test = [Y[i] for i in test_indices]

    return (X_train, Y_train, X_val, Y_val, X_test, Y_test)
        
class Vocabulary:
    """Frequency-based vocabulary for tokenized text."""

    def __init__(
            self, 
            vocab_size: int = 5000, 
            max_length: int = 100) -> None:

        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2.")

        if max_length <= 0:
            raise ValueError("max_length must be greater than zero.")

        self.vocab_size = vocab_size

        self.pad_token = "<pad>"
        self.unk_token = "<unk>"

        self.pad_id = 0
        self.unk_id = 1

        self.word_to_id: Dict[str, int] = {
            self.pad_token: self.pad_id,
            self.unk_token: self.unk_id,
        }

        self.token_counts: CounterType[str] = Counter()
        self.is_built = False

        self.max_length = max_length

    def build(
        self,
        tokenized_texts: List[List[str]],
    ) -> None:
        """
        Build a frequency-based vocabulary from tokenized training texts.

        Tokens are sorted by descending frequency. Tokens with equal
        frequencies are sorted alphabetically.

        Args:
            tokenized_texts: Tokenized training samples.
        """

        self.token_counts = Counter(
            token
            for sentence in tokenized_texts
            for token in sentence
        )

        ranked_tokens = sorted(
            self.token_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )

        number_of_special_tokens = 2
        maximum_regular_tokens = (
            self.vocab_size - number_of_special_tokens
        )

        most_frequent_tokens = ranked_tokens[
            :maximum_regular_tokens
        ]

        # Reset the vocabulary before rebuilding it.
        self.word_to_id = {
            self.pad_token: self.pad_id,
            self.unk_token: self.unk_id,
        }

        for token, _ in most_frequent_tokens:
            self.word_to_id[token] = len(self.word_to_id)

        self.is_built = True
        
    def encode(self, tokens: List[str]) -> List[int]:
        """
        Convert tokens into IDs using the constructed vocabulary.

        Tokens not present in the vocabulary are mapped to <unk>.
        """

        if not self.is_built:
            raise RuntimeError(
                "The vocabulary must be built before encoding tokens."
            )

        return [
            self.word_to_id.get(token, self.unk_id)
            for token in tokens
        ]

    def pad(self, token_ids: List[int]) -> List[int]:
        """
        Pad or truncate an encoded sequence to the configured maximum length.
        """
        token_ids = token_ids[:self.max_length]

        padding_length = self.max_length - len(token_ids)

        return token_ids + [self.pad_id] * padding_length

    def encode_and_pad_batch(
        self,
        tokenized_texts: List[List[str]],
    ) -> List[List[int]]:
        """Encode and pad multiple tokenized text samples."""
        
        return [
            self.pad(self.encode(tokens))
            for tokens in tokenized_texts
        ]

    def __len__(self) -> int:
        """Return the current vocabulary size."""

        return len(self.word_to_id)
