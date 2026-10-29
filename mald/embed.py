"""sentence embeddings, memoized per process.

minilm: matcher of the issue metrics and grouping of the real weaknesses.
bge:    retrieval of neighbor papers and clustering of the panel's issues.
models load from the local hf cache only (no download at run time).
"""
import os
from functools import cache

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
NAMES = {"minilm": "sentence-transformers/all-MiniLM-L6-v2", "bge": "BAAI/bge-large-en-v1.5"}


@cache
def model(name):
    import torch
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(NAMES[name], device="cuda" if torch.cuda.is_available() else "cpu")


_memo = {}


def vecs(texts, name="bge"):
    """unit-norm embeddings, shape (n, d). each text is encoded once per process."""
    if not texts:
        return np.zeros((0, 1024 if name == "bge" else 384), dtype=np.float32)
    new = list(dict.fromkeys(t for t in texts if (name, t) not in _memo))
    if new:
        for t, v in zip(new, model(name).encode(new, normalize_embeddings=True, batch_size=64)):
            _memo[(name, t)] = v
    return np.stack([_memo[(name, t)] for t in texts])
