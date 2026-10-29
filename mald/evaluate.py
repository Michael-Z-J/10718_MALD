"""score system outputs against gt: curves first, one pooled summary line.

usage: python -m mald.evaluate <split> <judge|nojudge> <name> [<name> ...]
  out/<split>/<name>.jsonl + out/gt_<split>.jsonl
  --> out/<split>/eval_<name>.json   (per-paper counts for every (c, tau) cell)
  --> out/<split>/summary.json       (merged across calls)

issue metrics are counted for every gt grouping threshold c in CGRID and
matching threshold tau in metrics.TAUS. reported per system:
  curve_c[m][c]   = population value at c, pooled over papers and all tau
  curve_tau[m][t] = population value at tau, pooled over papers and all c
  grid[m][c][t]   = population value of each cell
  m               = population value over papers and all cells (summary line)
population value = sum of numerators / sum of denominators (ndcg: mean over
(paper, cell) lists). bootstrap: resample papers, recompute the pooled value.
"""
import json, sys

import numpy as np

from mald import data, jl, metrics as M
from mald.paths import OUT
CGRID = [round(0.30 + 0.05 * i, 2) for i in range(11)]
POOLED = ["recall_union", "recall_multi", "recall_w", "precision", "ndcg_cons", "ndcg_def", "recall_at_1",
          "recall_at_3", "recall_at_5", "recall_at_10", "recall_at_20", "resolved_recall", "resolved_flagged"]
CURVED = ["recall_w", "recall_multi", "ndcg_cons", "precision"]
PANEL = ["alloc_best", "alloc_worst", "alloc_rand", "panel_union", "score_mae_best", "score_mae_worst"]
BOOT = ["ndcg_cons", "recall_w", "recall_multi", "precision", "resolved_recall"]


def load(p):
    return {r["forum"]: r for r in jl.rows(p)}


def per_paper(split, name, gts, use_judge):
    recs = load(OUT / split / f"{name}.jsonl")
    acc = dict(zip(data.forums()["forum"], data.forums()["accepted"]))
    rows = []
    for fid, rec in recs.items():
        if fid not in gts:
            continue
        g = gts[fid]
        rows.append(dict(forum=fid, accepted=bool(acc[fid]), **M.meta_level(g, rec, CGRID, use_judge),
                         **M.panel_level(g, rec), scores=M.paper_scores(g, rec, acc[fid])))
    return rows


def lift(papers, src="cells"):
    return M.pool(papers, "match_if_relevant", src=src) - M.pool(papers, "match_if_irrelevant", src=src)


def summarize(rows):
    s = {m: M.pool(rows, m) for m in POOLED}
    s["flag_lift"] = lift(rows)
    s["curve_c"] = {m: [M.pool(rows, m, {f"{c}|{t}" for t in M.TAUS}) for c in CGRID] for m in CURVED}
    s["curve_tau"] = {m: [M.pool(rows, m, {f"{c}|{t}" for c in CGRID}) for t in M.TAUS] for m in CURVED}
    s["grid"] = {m: [[M.pool(rows, m, {f"{c}|{t}"}) for t in M.TAUS] for c in CGRID] for m in CURVED}
    if any(r["judge"] for r in rows):
        s |= {f"j_{m}": M.pool(rows, m, src="judge") for m in POOLED}
        s["j_flag_lift"] = lift(rows, "judge")
    for k in PANEL + ["p5_legacy", "n_issues"]:
        v = [r[k] for r in rows if r.get(k) is not None]
        s[k] = float(np.nanmean(v)) if v else float("nan")
    s |= M.score_summary([r["scores"] for r in rows])
    for grp, flag in (("acc", True), ("rej", False)):
        sub = [r for r in rows if r["accepted"] == flag]
        s |= {f"{m}_{grp}": M.pool(sub, m) for m in ("recall_w", "ndcg_cons", "precision")}
        s[f"n_{grp}"] = len(sub)
    s["n_papers"] = len(rows)
    return s


def sums(rows, m):
    """per-paper (numerator, denominator) summed over all cells: arrays."""
    nd = np.array([[sum(c[m][0] for c in r["cells"].values() if m in c and not np.isnan(c[m][0])),
                    sum(c[m][1] for c in r["cells"].values() if m in c and not np.isnan(c[m][0]))] for r in rows])
    return nd[:, 0], nd[:, 1]


def boot(a, b, m, n=2000, seed=0):
    """paired bootstrap over papers of pooled(a) - pooled(b): (diff, lo, hi, P(diff <= 0))."""
    rng = np.random.default_rng(seed)
    (an, ad), (bn, bd) = sums(a, m), sums(b, m)
    idx = rng.integers(0, len(a), (n, len(a)))
    with np.errstate(invalid="ignore", divide="ignore"):
        ds = an[idx].sum(1) / ad[idx].sum(1) - bn[idx].sum(1) / bd[idx].sum(1)
        d0 = an.sum() / ad.sum() - bn.sum() / bd.sum()
    return [float(d0), float(np.nanquantile(ds, 0.025)), float(np.nanquantile(ds, 0.975)), float(np.nanmean(ds <= 0))]


def main(split, judge, *names):
    gts = load(OUT / f"gt_{split}.jsonl")
    summ, per = {}, {}
    for n in names:
        if not (OUT / split / f"{n}.jsonl").exists():
            continue
        rows = per_paper(split, n, gts, judge == "judge")
        if not rows:
            continue
        (OUT / split / f"eval_{n}.json").write_text(json.dumps(rows, default=float))
        per[n] = rows
        summ[n] = summarize(rows)
    if "b1" in per:
        ref = {r["forum"]: r for r in per["b1"]}
        for n, rows in per.items():
            if n == "b1":
                continue
            a = [r for r in rows if r["forum"] in ref]
            b = [ref[r["forum"]] for r in a]
            for m in BOOT:
                summ[n][f"boot_{m}_vs_b1"] = boot(a, b, m)
    path = OUT / split / "summary.json"
    allsys = json.loads(path.read_text())["systems"] if path.exists() else {}
    allsys.update(summ)
    path.write_text(json.dumps(dict(cgrid=CGRID, taus=M.TAUS, systems=allsys), indent=1, default=float))
    keys = ["n_papers", "recall_w", "recall_multi", "ndcg_cons", "precision", "resolved_recall", "flag_lift",
            "p5_legacy", "alloc_best", "alloc_worst", "panel_overall_spearman", "panel_accept_auc"]
    print("system".ljust(14) + "".join(k[:10].rjust(11) for k in keys))
    for n, s in summ.items():
        print(n.ljust(14) + "".join(f"{s.get(k, float('nan')):11.3f}" for k in keys))


if __name__ == "__main__":
    main(*sys.argv[1:])
