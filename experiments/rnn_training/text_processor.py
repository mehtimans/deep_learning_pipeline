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

def build_vocabulary(
        tokenized_text: List[List[str]], vocab_size: int = 5000,
) -> Tuple[Dict[str, int], CounterType[str]]:
    """
    Build a frequency-based vocabulary from tokenized training texts.

    The padding token is assigned ID 0, and the unknown token is
    assigned ID 1. The remaining IDs are assigned to the most
    frequent tokens.

    Args:
        tokenized_texts: Tokenized training samples.
        vocab_size: Maximum total vocabulary size, including special tokens.

    Returns:
        word_to_id: Mapping from tokens to integer IDs.
        token_counts: Token-frequency counter.
    """

    if vocab_size < 2 : 
        raise ValueError("vocab_size must be at least 2.")

    token_counts = Counter(
        token 
        for sentence in tokenized_text
        for token in sentence
    )

    # Sort by descending frequency and then alphabetically.
    ranked_tokens = sorted(
        token_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )

    most_frequent_tokens = ranked_tokens[:vocab_size - 2]

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    PAD_ID = 0
    UNK_ID = 1

    word_to_id = {
        PAD_TOKEN: PAD_ID,
        UNK_TOKEN: UNK_ID
    }
    
    for token, _ in most_frequent_tokens:
        word_to_id[token] = len(word_to_id)

    return word_to_id, token_counts
    






