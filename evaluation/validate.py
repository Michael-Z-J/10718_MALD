"""Validate the matcher against human labels.

1) sample: draw (issue, weakness) pairs from ALL systems' outputs (the embedding shortlist,
   so pairs are a mix of likely matches and near-misses) into a blind labelling sheet.
     python validate.py sample outputs/*.jsonl --n 200
   -> data/labels/template.csv  (copy to labels_<name>.csv; each annotator fills `label` 0/1/2
      independently, same scale as the judge)
   -> data/labels/key.jsonl     (pair_id -> system; annotators shouldn't look at it)

2) score: inter-annotator Cohen's kappa, then judge and embedding-threshold agreement with the
   majority label. Thresholds are picked on the tune half and reported on the test half.
     python validate.py score data/labels/labels_*.csv
"""

import argparse
import csv
import hashlib
import json
import os
import random
from itertools import combinations

import numpy as np
from sklearn.metrics import cohen_kappa_score

from common import DATA_DIR, embed, read_jsonl
from match import judge, load_reference

LABEL_DIR = os.path.join(DATA_DIR, "labels")
FIELDS = ["pair_id", "abstract", "issue", "weakness", "label"]


def sample(args):
    weaknesses_by_paper, abstracts = load_reference(substantive_only=False)
    pool = []
    for path in args.systems:
        system = os.path.splitext(os.path.basename(path))[0]
        for out in read_jsonl(path):
            ws = weaknesses_by_paper.get(out["paper_id"])
            if not ws or not out["issues"]:
                continue
            sims = embed(out["issues"]) @ embed([w["weakness"] for w in ws]).T
            for i, issue in enumerate(out["issues"]):
                for j in np.argsort(-sims[i])[:args.shortlist]:
                    pool.append({"system": system, "paper_id": out["paper_id"], "issue": issue,
                                 "weakness": ws[j]["weakness"], "sim": float(sims[i, j])})

    random.Random(args.seed).shuffle(pool)
    per_system = args.n // len(args.systems)  # balance systems so the matcher isn't tuned to one
    picked, counts = [], {}
    for p in pool:
        if counts.get(p["system"], 0) < per_system:
            counts[p["system"]] = counts.get(p["system"], 0) + 1
            picked.append(p)

    os.makedirs(LABEL_DIR, exist_ok=True)
    with open(os.path.join(LABEL_DIR, "template.csv"), "w", newline="") as f, \
         open(os.path.join(LABEL_DIR, "key.jsonl"), "w") as kf:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for k, p in enumerate(picked):
            w.writerow({"pair_id": k, "abstract": abstracts[p["paper_id"]],
                        "issue": p["issue"], "weakness": p["weakness"], "label": ""})
            kf.write(json.dumps({"pair_id": k, **p}) + "\n")
    print(f"Wrote {len(picked)} pairs ({counts}) -> {LABEL_DIR}")


def score(args):
    sheets = {}
    for path in args.labels:
        with open(path) as f:
            sheets[os.path.basename(path)] = {int(r["pair_id"]): r for r in csv.DictReader(f) if r["label"].strip()}
    ids = sorted(set.intersection(*(set(s) for s in sheets.values())))
    rows = next(iter(sheets.values()))
    binar = lambda x: int(int(x) >= args.min_score)
    human = {name: np.array([binar(s[i]["label"]) for i in ids]) for name, s in sheets.items()}

    print(f"{len(ids)} pairs labelled by all {len(sheets)} annotators (match = label >= {args.min_score})")
    for a, b in combinations(human, 2):
        print(f"  kappa {a} vs {b}: {cohen_kappa_score(human[a], human[b]):.3f}")
    majority = (np.mean(list(human.values()), axis=0) > 0.5).astype(int)

    sim = {p["pair_id"]: p["sim"] for p in read_jsonl(os.path.join(LABEL_DIR, "key.jsonl"))}
    sims = np.array([sim[i] for i in ids])
    judged = np.array([binar(judge(rows[i]["abstract"], rows[i]["issue"], rows[i]["weakness"])) for i in ids])
    test = np.array([int(hashlib.sha1(str(i).encode()).hexdigest(), 16) % 2 for i in ids], dtype=bool)

    thresholds = np.round(np.arange(0.3, 0.95, 0.05), 2)
    best_t = max(thresholds, key=lambda t: cohen_kappa_score(majority[~test], (sims[~test] >= t).astype(int)))
    for name, mask in (("tune", ~test), ("test", test)):
        print(f"[{name}, n={mask.sum()}] judge kappa={cohen_kappa_score(majority[mask], judged[mask]):.3f} "
              f"acc={np.mean(majority[mask] == judged[mask]):.3f} | "
              f"embedding@{best_t} kappa={cohen_kappa_score(majority[mask], (sims[mask] >= best_t).astype(int)):.3f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("systems", nargs="+")
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--shortlist", type=int, default=3)
    s.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("score")
    c.add_argument("labels", nargs="+")
    c.add_argument("--min-score", type=int, default=2)
    args = ap.parse_args()
    sample(args) if args.cmd == "sample" else score(args)


if __name__ == "__main__":
    main()
