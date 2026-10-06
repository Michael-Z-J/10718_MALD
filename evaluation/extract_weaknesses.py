"""Stage 1: turn each review's `weaknesses` field into atomic weaknesses, then
deduplicate across the reviewers of each paper.

Writes data/weaknesses.jsonl, one row per deduplicated weakness:
  {paper_id, weakness_id, weakness, type, n_reviewers, reviewers, variants}

Usage:
  python extract_weaknesses.py                  # LLM split (needs call_llm)
  python extract_weaknesses.py --method regex   # bullet split only, type="unknown"
"""

import argparse
import os
import re
from collections import defaultdict

import numpy as np

from common import DATA_DIR, call_llm, embed, parse_json, read_jsonl, write_jsonl

EXTRACT_PROMPT = """Below is the "Weaknesses" section of a peer review of an ML paper. List every distinct weakness or criticism
the reviewer raises about the paper. Rules:
- One weakness per item, one sentence each, preserving specifics (method names,
  datasets, section/table numbers).
- Exclude praise, summaries of the paper, and questions that aren't criticisms.
- Tag each item: "substantive" (experiments, baselines, claims, theory) or
  "minor" (clarity, typos, formatting).
Return JSON: [{{"weakness": "...", "type": "substantive|minor"}}]

Review:
{review_text}"""

BULLET = re.compile(r"^\s*(?:[-*•]|\(?\d+[.)]|\(?[a-z][.)]|W\d+[.:)]?)\s+", re.M)


def split_regex(text):
    """Split on bullet/numbered-list markers; fall back to the whole text as one item."""
    parts = [p.strip() for p in BULLET.split(text) if p and p.strip()]
    return [{"weakness": " ".join(p.split()), "type": "unknown"} for p in parts if len(p) > 20]


def split_llm(text):
    return parse_json(call_llm(EXTRACT_PROMPT.format(review_text=text)))


def dedup(items, threshold, vecs=None):
    """Greedy clustering: an item joins the most similar existing cluster if cosine >= threshold
    and that cluster doesn't already contain this reviewer. Returns lists of items."""
    if not items:
        return []
    if vecs is None:
        vecs = embed([it["weakness"] for it in items])
    clusters = []  # (centroid_index, [item indices], reviewer set)
    for i, it in enumerate(items):
        best, best_sim = None, threshold
        for c in clusters:
            sim = float(vecs[i] @ vecs[c[0]])
            if sim >= best_sim and it["reviewer"] not in c[2]:
                best, best_sim = c, sim
        if best is None:
            clusters.append((i, [i], {it["reviewer"]}))
        else:
            best[1].append(i)
            best[2].add(it["reviewer"])
    return [[items[j] for j in c[1]] for c in clusters]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["llm", "regex"], default="llm")
    ap.add_argument("--dedup-threshold", type=float, default=0.8)
    args = ap.parse_args()

    split = split_llm if args.method == "llm" else split_regex
    by_paper = defaultdict(list)
    for r in read_jsonl(os.path.join(DATA_DIR, "reviews.jsonl")):
        if r["weaknesses"].strip():
            for it in split(r["weaknesses"]):
                by_paper[r["paper_id"]].append({**it, "reviewer": r["reviewer"]})

    # Embed everything in one batched pass; per-paper calls are dominated by overhead.
    texts = sorted({it["weakness"] for items in by_paper.values() for it in items})
    vectors = dict(zip(texts, embed(texts, batch_size=512, show_progress_bar=True)))

    rows = []
    for pid, items in by_paper.items():
        vecs = np.stack([vectors[it["weakness"]] for it in items])
        for k, cluster in enumerate(dedup(items, args.dedup_threshold, vecs)):
            types = {it["type"] for it in cluster}
            rows.append({
                "paper_id": pid,
                "weakness_id": f"{pid}#{k}",
                "weakness": cluster[0]["weakness"],
                "type": "substantive" if "substantive" in types else cluster[0]["type"],
                "n_reviewers": len({it["reviewer"] for it in cluster}),
                "reviewers": sorted({it["reviewer"] for it in cluster}),
                "variants": [it["weakness"] for it in cluster[1:]],
            })
    write_jsonl(os.path.join(DATA_DIR, "weaknesses.jsonl"), rows)
    print(f"{len(rows)} weaknesses over {len(by_paper)} papers")


if __name__ == "__main__":
    main()
