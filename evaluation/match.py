"""Stages 2-3: match a system's issues to reviewer weaknesses and score recall@k / precision@k.

System output format (one JSONL file per system, identical for every baseline and our model):
  {"paper_id": "...", "issues": ["issue 1", ..., "issue 5"]}

Matching: embeddings shortlist the top `--shortlist` weaknesses per issue, a blind LLM judge
scores each candidate pair 0/1/2, then a one-to-one (Hungarian) assignment over pairs with
score >= `--min-score` decides matches, so one broad issue can't claim several weaknesses.

Usage:
  python match.py outputs/qwen.jsonl outputs/nonml.jsonl --substantive-only
"""

import argparse
import hashlib
import json
import os
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from common import DATA_DIR, call_llm, embed, parse_json, read_jsonl

JUDGE_PROMPT = """You are checking whether an automatically generated critique of a paper raises the same concern
as a criticism written by a human peer reviewer.

Paper abstract:
{abstract}

Critique A: {issue}
Critique B: {weakness}

Score:
2 = same concern about the same aspect of the paper (e.g. both ask for the same missing baseline).
1 = partially overlapping: related concern, but one is clearly broader or targets a different detail.
0 = different concerns.
A generic critique (e.g. "experiments are insufficient") must NOT get 2 against a specific one
(e.g. "no comparison to method X on dataset Y").
Return JSON: {{"score": 0|1|2, "reason": "..."}}"""

CACHE_PATH = os.path.join(DATA_DIR, "judge_cache.jsonl")
_cache = None


def judge(abstract, issue, weakness):
    """Cached LLM judge call. The judge never sees which system produced `issue`."""
    global _cache
    if _cache is None:
        _cache = {r["key"]: r["score"] for r in read_jsonl(CACHE_PATH)} if os.path.exists(CACHE_PATH) else {}
    key = hashlib.sha1(json.dumps([abstract, issue, weakness]).encode()).hexdigest()
    if key not in _cache:
        out = parse_json(call_llm(JUDGE_PROMPT.format(abstract=abstract, issue=issue, weakness=weakness)))
        _cache[key] = int(out["score"])
        with open(CACHE_PATH, "a") as f:
            f.write(json.dumps({"key": key, "score": _cache[key], "reason": out.get("reason")}) + "\n")
    return _cache[key]


def match_paper(abstract, issues, weaknesses, shortlist, min_score):
    """Returns (score matrix [issues x weaknesses], list of matched (issue_idx, weakness_idx))."""
    scores = np.zeros((len(issues), len(weaknesses)))
    if not issues or not weaknesses:
        return scores, []
    sims = embed(issues) @ embed([w["weakness"] for w in weaknesses]).T
    for i in range(len(issues)):
        for j in np.argsort(-sims[i])[:shortlist]:
            scores[i, j] = judge(abstract, issues[i], weaknesses[j]["weakness"])
    rows, cols = linear_sum_assignment(-scores)
    return scores, [(i, j) for i, j in zip(rows, cols) if scores[i, j] >= min_score]


def evaluate(system_path, weaknesses_by_paper, abstracts, args):
    recall, w_recall, precision = [], [], []
    for out in read_jsonl(system_path):
        ws = weaknesses_by_paper.get(out["paper_id"])
        if not ws:
            continue
        issues = out["issues"][:args.k]
        _, matches = match_paper(abstracts[out["paper_id"]], issues, ws, args.shortlist, args.min_score)
        hit = {j for _, j in matches}
        recall.append(len(hit) / len(ws))
        w_recall.append(sum(ws[j]["n_reviewers"] for j in hit) / sum(w["n_reviewers"] for w in ws))
        precision.append(len(matches) / max(len(issues), 1))
    return {"n_papers": len(recall), f"recall@{args.k}": np.mean(recall),
            f"weighted_recall@{args.k}": np.mean(w_recall), f"precision@{args.k}": np.mean(precision)}


def load_reference(substantive_only):
    by_paper = defaultdict(list)
    for w in read_jsonl(os.path.join(DATA_DIR, "weaknesses.jsonl")):
        if not substantive_only or w["type"] != "minor":
            by_paper[w["paper_id"]].append(w)
    abstracts = {p["paper_id"]: p["abstract"] for p in read_jsonl(os.path.join(DATA_DIR, "papers.jsonl"))}
    return by_paper, abstracts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="+", help="system output JSONL files")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--shortlist", type=int, default=3, help="judge calls per issue")
    ap.add_argument("--min-score", type=int, default=2, help="judge score that counts as a match")
    ap.add_argument("--substantive-only", action="store_true", help="drop clarity/typo weaknesses")
    args = ap.parse_args()

    weaknesses_by_paper, abstracts = load_reference(args.substantive_only)
    for path in args.systems:
        res = evaluate(path, weaknesses_by_paper, abstracts, args)
        print(os.path.basename(path), {k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})


if __name__ == "__main__":
    main()
