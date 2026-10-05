"""Shared helpers: the LLM stub, sentence embeddings, and JSONL I/O."""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")


def call_llm(prompt: str) -> str:
    """Send `prompt` to the extractor/judge LLM and return its text response.

    TODO: plug in a model here. Use a different model family from the LLM baseline
    being evaluated, since judges favour text written in their own style.
    """
    raise NotImplementedError("Fill in call_llm() in evaluation/common.py")


def parse_json(text: str):
    """Pull the first JSON array/object out of an LLM response."""
    start = min((i for i in (text.find("["), text.find("{")) if i != -1), default=-1)
    if start == -1:
        raise ValueError(f"No JSON in LLM response: {text[:200]!r}")
    end = max(text.rfind("]"), text.rfind("}"))
    return json.loads(text[start:end + 1])


_embedder = None


def embed(texts):
    """L2-normalised sentence embeddings, so a dot product is cosine similarity."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return np.asarray(_embedder.encode(list(texts), normalize_embeddings=True))


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
