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
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

from collections import Counter
from typing import Counter as CounterType
from typing import Dict, List, Tuple


class Vocabulary:
    """Frequency-based vocabulary for tokenized text."""

    def __init__(self, vocab_size: int = 5000) -> None:
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2.")

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

        Returns:
            word_to_id: Mapping from tokens to integer IDs.
            token_counts: Frequency count for every training token.
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

    def encode_batch(
        self,
        tokenized_texts: List[List[str]],
    ) -> List[List[int]]:
        """Encode multiple tokenized text samples."""

        return [
            self.encode(tokens)
            for tokens in tokenized_texts
        ]

    def __len__(self) -> int:
        """Return the current vocabulary size."""

        return len(self.word_to_id)