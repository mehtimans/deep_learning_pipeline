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
from typing import Tuple, List

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