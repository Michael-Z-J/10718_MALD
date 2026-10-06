"""Stages 2-3: match a system's issues to reviewer weaknesses and score recall@k / precision@k.

System output format (one JSONL file per system, identical for every baseline and our model):
  {"paper_id": "...", "issues": ["issue 1", ..., "issue 5"]}

Matching: embeddings shortlist the top `--shortlist` weaknesses per issue, a blind LLM judge
scores each candidate pair 0/1/2, then a one-to-one (Hungarian) assignment over pairs with
score >= `--min-score` decides matches, so one broad issue can't claim several weaknesses.
Reports precision@k (the headline metric), recall@k, recall@k weighted by n_reviewers and by
reviewer severity (see load_reference), and the oracle recall ceiling min(k, |W|) / |W|.
Systems with ties at the top-k cutoff are scored by the expected value over tie-breaks (issue_sets).
Two scores are normalized to the k-issue capacity, so 1.0 is reachable on every paper:
  normalized_recall@k          = matches / min(k, |W|)
  normalized_severity_recall@k = severity of matched weaknesses / severity of the k most severe,
                                 i.e. 1.0 iff the k issues match the k most severe weaknesses.
`--matcher embedding` skips the judge: a pair matches when its sentence-embedding cosine is
>= the threshold (one result per value of `--sim-threshold`), still one-to-one.

Usage:
  python match.py outputs/qwen.jsonl outputs/nonml.jsonl --substantive-only
"""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from itertools import combinations

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


_vectors = {}


def precompute_embeddings(texts):
    """Embed every text once in large batches, so per-paper matching is just a lookup."""
    new = sorted(set(texts) - set(_vectors))
    _vectors.update(zip(new, embed(new, batch_size=512, show_progress_bar=True)))


def match_embedding(issues, weaknesses, threshold):
    """One-to-one matches (issue_idx, weakness_idx) between pairs with cosine >= threshold."""
    if not issues or not weaknesses:
        return []
    sims = np.stack([_vectors[t] for t in issues]) @ np.stack([_vectors[w["weakness"]] for w in weaknesses]).T
    rows, cols = linear_sum_assignment(-(sims >= threshold).astype(float))
    return [(i, j) for i, j in zip(rows, cols) if sims[i, j] >= threshold]


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


def issue_sets(out, k):
    """The equally likely top-k issue lists for one system output. A system whose ranking has ties
    at the cutoff writes "issue_ties": {"fixed": [...], "pool": [...], "k": m}: every m-subset of
    the tied pool is equally likely, so we enumerate them all (exact expectation over random
    tie-breaking). Without ties there is just one list."""
    ties = out.get("issue_ties")
    if not ties or len(ties["pool"]) <= ties["k"]:
        return [out["issues"][:k]]
    return [ties["fixed"] + list(c) for c in combinations(ties["pool"], ties["k"])]


def paper_metrics(issues, ws, abstract, args, threshold):
    if args.matcher == "embedding":
        matches = match_embedding(issues, ws, threshold)
    else:
        _, matches = match_paper(abstract, issues, ws, args.shortlist, args.min_score)
    hit = {j for _, j in matches}
    top_k = sorted((w["severity"] for w in ws), reverse=True)[:args.k]
    return {f"precision@{args.k}": len(matches) / max(len(issues), 1),
            f"recall@{args.k}": len(hit) / len(ws),
            f"weighted_recall@{args.k}": sum(ws[j]["n_reviewers"] for j in hit) / sum(w["n_reviewers"] for w in ws),
            f"severity_recall@{args.k}": sum(ws[j]["severity"] for j in hit) / sum(w["severity"] for w in ws),
            f"normalized_recall@{args.k}": len(hit) / len(top_k),
            f"normalized_severity_recall@{args.k}": sum(ws[j]["severity"] for j in hit) / sum(top_k),
            f"oracle_recall@{args.k}": min(args.k, len(ws)) / len(ws)}


def evaluate(system_path, weaknesses_by_paper, abstracts, args, threshold=None):
    """Mean over papers of each metric; a paper with ties at the cutoff contributes the expected
    value of each metric over its equally likely top-k sets."""
    per_paper, n_tied = [], 0
    for out in read_jsonl(system_path):
        ws = weaknesses_by_paper.get(out["paper_id"])
        if not ws or (args.paper_ids is not None and out["paper_id"] not in args.paper_ids):
            continue
        sets = issue_sets(out, args.k)
        n_tied += len(sets) > 1
        draws = [paper_metrics(issues, ws, abstracts.get(out["paper_id"]), args, threshold) for issues in sets]
        per_paper.append({m: np.mean([d[m] for d in draws]) for m in draws[0]})
    res = {"n_papers": len(per_paper), "n_papers_with_ties": n_tied}
    res.update({m: float(np.mean([p[m] for p in per_paper])) for m in per_paper[0]})
    return res


def rating_value(rating):
    """OpenReview ratings are ints (2026) or strings like "5: marginally below ..." (2024)."""
    return int(str(rating).split(":")[0])


def load_reference(substantive_only):
    """Weaknesses per paper. Each gets a `severity` weight: the sum over the reviewers who raised
    it of (11 - their rating), so a weakness from a strong-reject (0) reviewer counts 11 and one
    from a strong-accept (10) reviewer counts 1. Ratings are per review, not per weakness."""
    ratings = {(r["paper_id"], r["reviewer"]): rating_value(r["rating"])
               for r in read_jsonl(os.path.join(DATA_DIR, "reviews.jsonl")) if r["rating"] is not None}
    by_paper = defaultdict(list)
    for w in read_jsonl(os.path.join(DATA_DIR, "weaknesses.jsonl")):
        if not substantive_only or w["type"] != "minor":
            w["severity"] = sum(11 - ratings[(w["paper_id"], r)] for r in w["reviewers"])
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
    ap.add_argument("--matcher", choices=["llm", "embedding"], default="llm")
    ap.add_argument("--sim-threshold", type=float, nargs="+", default=[0.5, 0.6, 0.7],
                    help="cosine threshold(s) for --matcher embedding")
    ap.add_argument("--paper-ids", default=None,
                    help="JSON list of paper_ids to score (e.g. data/sample_2026_500.json from fetch_pdfs.py)")
    args = ap.parse_args()
    if args.paper_ids:
        with open(args.paper_ids) as f:
            args.paper_ids = set(json.load(f))

    weaknesses_by_paper, abstracts = load_reference(args.substantive_only)
    thresholds = args.sim_threshold if args.matcher == "embedding" else [None]
    for path in args.systems:
        if args.matcher == "embedding":
            outs = [o for o in read_jsonl(path) if weaknesses_by_paper.get(o["paper_id"])
                    and (args.paper_ids is None or o["paper_id"] in args.paper_ids)]
            precompute_embeddings([t for o in outs for issues in issue_sets(o, args.k) for t in issues] +
                                  [w["weakness"] for o in outs for w in weaknesses_by_paper[o["paper_id"]]])
        for t in thresholds:
            res = evaluate(path, weaknesses_by_paper, abstracts, args, t)
            name = os.path.basename(path) + (f" [cos>={t}]" if t is not None else "")
            print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})


if __name__ == "__main__":
    main()
