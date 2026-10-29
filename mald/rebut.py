"""confusion matrix of the "doable" flag against rebuttal ground truth.

gt concern labels (from each reviewer's own follow-up after the rebuttal):
  rebuttable     = the reviewer called it resolved
  non-rebuttable = the reviewer said it is still open (and nobody called it resolved)
  others (no follow-up statement) are left out.
for every (gt grouping c, matching tau) cell, each labelled concern falls in one
column: not found by the meta-review / found and flagged doable / found and not
flagged doable ("doable" also requires "relevant" when the system flags both).
counts are averaged over cells (expected counts per cell); rates are pooled.
summary per system:
  hit_reb, hit_non   = share of rebuttable / non-rebuttable concerns found
  flag_reb, flag_non = among found, share flagged doable
  gap                = flag_reb - flag_non (> 0: the flag separates the two)
paired bootstrap over papers for gap.

usage: python -m mald.rebut <split> <name=label> [...] > table.tex
"""
import json, sys

import numpy as np

from mald import gt as G, jl, metrics as M
from mald.evaluate import CGRID, OUT

COLS = ("miss", "flag", "noflag")


def flagged(x):
    return bool(x.get("doable")) and x.get("relevant") is not False


def counts(g, rec):
    """per-paper counts: {"reb"|"non": {col: total over cells}}, number of cells."""
    issues = rec["meta"]["issues"]
    out = {k: dict.fromkeys(COLS, 0.0) for k in ("reb", "non")}
    cells = 0
    for c in CGRID:
        cl = G.clusters(g, c)
        if not cl:
            continue
        S = M.sims(cl, issues)
        for t in M.TAUS:
            cells += 1
            for i, k in enumerate(cl):
                lab = "reb" if k["resolved"] else "non" if k["open"] else None
                if lab is None:
                    continue
                hit = np.nonzero(S[i] >= t)[0] if len(issues) else []
                col = "miss" if not len(hit) else "flag" if any(flagged(issues[j]) for j in hit) else "noflag"
                out[lab][col] += 1
    return out, cells


def rates(per):
    tot = {k: {c: sum(p[0][k][c] for p in per) for c in COLS} for k in ("reb", "non")}
    n = {k: sum(tot[k].values()) for k in tot}
    found = {k: tot[k]["flag"] + tot[k]["noflag"] for k in tot}
    r = dict(hit_reb=found["reb"] / n["reb"] if n["reb"] else np.nan,
             hit_non=found["non"] / n["non"] if n["non"] else np.nan,
             flag_reb=tot["reb"]["flag"] / found["reb"] if found["reb"] else np.nan,
             flag_non=tot["non"]["flag"] / found["non"] if found["non"] else np.nan)
    r["gap"] = r["flag_reb"] - r["flag_non"]
    return r, tot


def boot(per, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    gs = [rates([per[i] for i in rng.integers(0, len(per), len(per))])[0]["gap"] for _ in range(n)]
    return float(np.nanquantile(gs, 0.025)), float(np.nanquantile(gs, 0.975))


def main(split, *pairs):
    gts = {r["forum"]: r for r in jl.rows(OUT / f"gt_{split}.jsonl")}
    res = {}
    for p in pairs:
        name, lab = p.split("=", 1)
        lab = lab.replace("_", " ")
        recs = {r["forum"]: r for r in jl.rows(OUT / split / f"{name}.jsonl")}
        per = [counts(gts[f], r) for f, r in recs.items() if f in gts]
        if not per:
            continue
        cells = per[0][1]
        r, tot = rates(per)
        r["ci"] = boot(per)
        r["mat"] = {k: {c: v / cells for c, v in tot[k].items()} for k in tot}
        res[lab] = r
    (OUT / split / "rebut.json").write_text(json.dumps(res, indent=1, default=float))
    f = lambda x: "n/a" if x != x else f"{x:.2f}"
    rows = [r"\begin{table}[H]\centering\small", r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{l|rrr|rrr|rrrr}", r"\toprule",
            r" & \multicolumn{3}{c|}{rebuttable (resolved)} & \multicolumn{3}{c|}{non-rebuttable (still open)} & & & & \\",
            r"system & missed & flagged & not fl. & missed & flagged & not fl. & found$_{reb}$ & found$_{non}$ & gap & 95\% CI \\",
            r"\midrule"]
    for lab, r in res.items():
        m = r["mat"]
        rows.append(f"{lab} & " + " & ".join(f(m["reb"][c]) for c in COLS) + " & "
                    + " & ".join(f(m["non"][c]) for c in COLS)
                    + f" & {f(r['hit_reb'])} & {f(r['hit_non'])} & {f(r['gap'])} & [{f(r['ci'][0])}, {f(r['ci'][1])}] \\\\")
    rows += [r"\bottomrule", r"\end{tabular}}",
             r"\caption{Doable flag against rebuttal ground truth (" + split + r"). Concerns are labelled from each real reviewer's own follow-up after the rebuttal, as rebuttable if the reviewer called them resolved and as non-rebuttable if still open. Counts are expected concerns per $(c,\tau)$ cell, summed over papers. found: share of each class that the meta-review states. gap: P(flagged doable $\mid$ found, rebuttable) $-$ P(flagged doable $\mid$ found, non-rebuttable), with a paired bootstrap over papers.}",
             r"\end{table}"]
    print("\n".join(rows))


if __name__ == "__main__":
    main(*sys.argv[1:])
