"""extra test tables from stored per-paper evaluation rows (no llm calls).

out/test/eval_<name>.json --> report/extra_tables.tex + out/test/extra.json
  1. found / missed confusion matrix of real concerns, split by how many real
     reviewers raised them (1 vs >= 2), and matched / unmatched system issues;
     expected counts per (c, tau) cell, summed over papers
  2. differences to the ml baseline (paired bootstrap over papers, seed 0)
  3. score agreement: spearman of panel-mean vs real-mean scores per rubric axis
     with a bootstrap 95% interval, accept auc, mean absolute error, and
     weak-axis agreement: share of papers where the panel and the real
     reviewers agree on which of soundness / contribution scored lower
     (papers with a real tie are left out)

  4. joint score agreement over the rubric axes that both sides rate on iclr 2023
     (overall, soundness, contribution):
     paper level: panel-mean vector vs real-mean vector across papers
     reviewer level: bots matched to real reviewers by the hungarian assignment
       on the l1 distance of their score vectors (best = min total distance,
       worst = max); all bot x reviewer pairs = random allocation
     statistics: pooled within-axis spearman (ranks taken per axis, then one
     correlation over all (unit, axis) entries) with bootstrap 95% intervals
     over papers, and distance correlation of the per-axis ranks with a
     permutation p-value (bootstrap duplicates bias distance correlation)
  5. the same score agreement split by the final decision (accepted / rejected),
     plus the mean signed error panel - real per axis (> 0: optimistic) with a
     bootstrap 95% interval over papers

usage: python -m mald.extra
"""
import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

from mald import jl, paths
from mald.evaluate import boot

OUT = paths.OUT / "test"
REP = paths.REPORT
NAMES = {"b0": "Non-ML baseline", "b1old": "ML baseline", "b1": "Ours", "m": "Ours (agentic)",
         "m_unc": "Ours (agentic), no rebuttal-mode", "m_alt": "Ours (agentic), LLM meta-review",
         "b1_nemotron": "Ours, Nemotron", "m_nemotron": "Ours (agentic), Nemotron", "m_mix2": "Ours (agentic), mixed panel"}
AXES = ("overall", "soundness", "contribution")


def rows(n):
    p = OUT / f"eval_{n}.json"
    return json.loads(p.read_text()) if p.exists() else []


def confusion(rs):
    tot = {k: 0.0 for k in ("hit_all", "n_all", "hit_multi", "n_multi", "matched", "n_issues")}
    cells = 0
    for r in rs:
        for c in r["cells"].values():
            cells += 1
            tot["hit_all"] += c["recall_union"][0]; tot["n_all"] += c["recall_union"][1]
            tot["hit_multi"] += c["recall_multi"][0]; tot["n_multi"] += c["recall_multi"][1]
            tot["matched"] += c["precision"][0]; tot["n_issues"] += c["precision"][1]
    k = cells / len(rs) if rs else 1
    t = {a: v / k for a, v in tot.items()}
    return dict(single_found=t["hit_all"] - t["hit_multi"], single_missed=(t["n_all"] - t["n_multi"]) - (t["hit_all"] - t["hit_multi"]),
                multi_found=t["hit_multi"], multi_missed=t["n_multi"] - t["hit_multi"],
                issues_matched=t["matched"], issues_unmatched=t["n_issues"] - t["matched"])


def spear_ci(x, y, n=2000, seed=0):
    x, y = np.asarray(x), np.asarray(y)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
            bs.append(spearmanr(x[i], y[i]).statistic)
    return float(spearmanr(x, y).statistic), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def scores(rs):
    sc = [r["scores"] for r in rs if r["scores"]["panel"]]
    if len(sc) < 5:
        return None
    out = {}
    for a in AXES:
        pr = [(s["panel"][a], s["real"][a]) for s in sc if s["panel"].get(a) is not None and s["real"][a] == s["real"][a]]
        x, y = zip(*pr)
        out[a] = spear_ci(x, y)
        out[f"{a}_mae"] = float(np.mean(np.abs(np.array(x) - np.array(y))))
    out["auc"] = float(roc_auc_score([s["accepted"] for s in sc], [s["panel"]["overall"] for s in sc]))
    agree = [np.sign(s["panel"]["soundness"] - s["panel"]["contribution"]) == np.sign(s["real"]["soundness"] - s["real"]["contribution"])
             for s in sc if s["real"]["soundness"] != s["real"]["contribution"]]
    out["weak_axis"] = (float(np.mean(agree)), len(agree))
    out["n"] = len(sc)
    return out


def bias(rs, n=2000, seed=0):
    """mean signed error panel - real per axis, with a bootstrap interval."""
    sc = [r["scores"] for r in rs if r["scores"]["panel"]]
    rng = np.random.default_rng(seed)
    out = {}
    for a in AXES:
        d = np.array([s["panel"][a] - s["real"][a] for s in sc
                      if s["panel"].get(a) is not None and s["real"][a] == s["real"][a]])
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
        out[a] = (float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)))
    return out


def pooled_rho(X, Y):
    """pooled within-axis spearman of two (units x axes) arrays."""
    rx = np.column_stack([rankdata(X[:, a]) for a in range(X.shape[1])])
    ry = np.column_stack([rankdata(Y[:, a]) for a in range(Y.shape[1])])
    rx, ry = rx - rx.mean(0), ry - ry.mean(0)
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def dcor(X, Y):
    """distance correlation of the per-axis ranks of two (units x axes) arrays."""
    def cen(Z):
        Z = np.column_stack([rankdata(Z[:, a]) for a in range(Z.shape[1])])
        d = np.linalg.norm(Z[:, None] - Z[None], axis=-1)
        return d - d.mean(0) - d.mean(1)[:, None] + d.mean()
    A, B = cen(X), cen(Y)
    v = (A * B).mean() / np.sqrt((A * A).mean() * (B * B).mean())
    return float(np.sqrt(max(v, 0)))


def vec(d):
    v = [d.get(a) for a in AXES]
    return None if any(x is None or x != x for x in v) else v


def joint(name, gts):
    """paper- and reviewer-level joint agreement with bootstrap over papers."""
    recs = {r["forum"]: r for r in jl.rows(OUT / f"{name}.jsonl")}
    papers = []
    for fid, g in sorted(gts.items()):
        r = recs.get(fid)
        if not r or not r["panel"]:
            continue
        bots = [v for v in (vec(b["scores"]) for b in r["panel"]) if v]
        real = [v for v in (vec(x["scores"]) for x in g["reviewers"]) if v]
        if not bots or not real:
            continue
        B, R = np.array(bots), np.array(real)
        cost = np.abs(B[:, None] - R[None]).sum(-1)
        rep = int(np.ceil(len(R) / len(B)))
        big = np.tile(cost, (rep, 1))
        bi, bj = linear_sum_assignment(big)
        wi, wj = linear_sum_assignment(-big)
        Bt = np.tile(B, (rep, 1))
        papers.append(dict(pm=B.mean(0), rm=R.mean(0), best=(Bt[bi], R[bj]), worst=(Bt[wi], R[wj]),
                           rand=(np.repeat(B, len(R), 0), np.tile(R, (len(B), 1)))))
    if len(papers) < 5:
        return None

    def stats(ps):
        out = {}
        X = np.array([p["pm"] for p in ps]); Y = np.array([p["rm"] for p in ps])
        out["paper_rho"], out["paper_dcor"] = pooled_rho(X, Y), dcor(X, Y)
        for k in ("best", "worst", "rand"):
            X = np.vstack([p[k][0] for p in ps]); Y = np.vstack([p[k][1] for p in ps])
            out[f"{k}_rho"] = pooled_rho(X, Y)
        return out

    est = stats(papers)
    rng = np.random.default_rng(0)
    bs = [stats([papers[i] for i in rng.integers(0, len(papers), len(papers))]) for _ in range(500)]
    out = {k: (v, float(np.quantile([b[k] for b in bs], 0.025)), float(np.quantile([b[k] for b in bs], 0.975))) for k, v in est.items()}
    # bootstrap duplicates bias distance correlation upward: use a permutation p-value
    X = np.array([p["pm"] for p in papers]); Y = np.array([p["rm"] for p in papers])
    perm = [dcor(X, Y[rng.permutation(len(Y))]) for _ in range(1000)]
    out["paper_dcor"] = (est["paper_dcor"], float(np.mean([q >= est["paper_dcor"] for q in perm])), None)
    return out | {"n": len(papers)}


def f(x):
    return "n/a" if x is None or x != x else f"{x:.2f}"


def main():
    res = {}
    gts = {g["forum"]: g for g in jl.rows(OUT.parent / "gt_test.jsonl")}
    base = {r["forum"]: r for r in rows("b1old")}
    for n in NAMES:
        rs = rows(n)
        if not rs:
            continue
        a = [r for r in rs if r["forum"] in base]
        b = [base[r["forum"]] for r in a]
        res[n] = dict(conf=confusion(rs), scores=scores(rs), joint=joint(n, {f: gts[f] for f in base}) if n != "b1old" else None,
                      boot={m: boot(a, b, m) for m in ("recall_w", "recall_multi", "ndcg_cons", "precision", "resolved_recall")} if n != "b1old" else {})
        acc = {r["forum"]: r["accepted"] for r in rs}
        res[n]["by_dec"] = {}
        for k, v in (("accepted", True), ("rejected", False)):
            sub = [r for r in rs if r["accepted"] == v]
            if n in ("b1old", "m_alt") or not scores(sub):
                continue
            res[n]["by_dec"][k] = dict(scores=scores(sub), bias=bias(sub),
                                       joint=joint(n, {f: gts[f] for f in base if acc.get(f) == v}))
    (OUT / "extra.json").write_text(json.dumps(res, indent=1, default=float))
    t = [r"\begin{table}[H]\centering\small", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r" & \multicolumn{2}{c}{raised by one reviewer} & \multicolumn{2}{c}{raised by $\ge$2 reviewers} & \multicolumn{2}{c}{system issues} \\",
         r"System & found & missed & found & missed & match a concern & match none \\", r"\midrule"]
    for n, r in res.items():
        c = r["conf"]
        t.append(f"{NAMES[n]} & {c['single_found']:.1f} & {c['single_missed']:.1f} & {c['multi_found']:.1f} & {c['multi_missed']:.1f} & {c['issues_matched']:.1f} & {c['issues_unmatched']:.1f} \\\\")
    t += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{Real reviewer concerns found and missed by each meta-review, and system issues that do or do not match a real concern, on the 40 test papers. Counts are summed over papers and averaged over the $(c,\tau)$ grid. The recall and precision figures in the main test table are ratios of these counts.}",
          r"\label{tab:confusion}", r"\end{table}"]
    t += [r"\begin{table}[H]\centering\small", r"\begin{tabular}{llrrr}", r"\toprule",
          r"System & Metric & Difference & 95\% CI low & 95\% CI high \\", r"\midrule"]
    lab = {"recall_w": "weighted recall", "recall_multi": "recall, shared concerns", "ndcg_cons": "nDCG", "precision": "precision", "resolved_recall": "recall, resolved concerns"}
    for n, r in res.items():
        for m, v in r["boot"].items():
            t.append(f"{NAMES[n]} & {lab[m]} & {f(v[0])} & {f(v[1])} & {f(v[2])} \\\\")
    t += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Difference to the ML baseline on the 40 test papers (pooled value of the system minus that of the ML baseline), with a paired bootstrap over papers (2{,}000 resamples).}",
          r"\label{tab:boot}", r"\end{table}"]
    t += [r"\begin{table}[H]\centering\small", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
          r"System & $\rho$ overall & $\rho$ soundness & $\rho$ contribution & accept AUC & MAE overall & weaker axis agrees & papers \\", r"\midrule"]
    for n, r in res.items():
        s = r["scores"]
        if not s:
            continue
        cell = lambda a: f"{f(s[a][0])} [{f(s[a][1])}, {f(s[a][2])}]"
        t.append(f"{NAMES[n]} & {cell('overall')} & {cell('soundness')} & {cell('contribution')} & {f(s['auc'])} & {f(s['overall_mae'])} & {f(s['weak_axis'][0])} ({s['weak_axis'][1]}) & {s['n']} \\\\")
    t += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{Agreement between the simulated panel's mean scores and the real reviewers' mean scores on the 40 test papers. $\rho$: Spearman correlation across papers with a bootstrap 95\% interval. Accept AUC: how well the panel's mean overall score separates accepted from rejected papers. Weaker axis agrees: share of papers on which the panel and the real reviewers agree on which of soundness and contribution scored lower (number of papers without a real tie in parentheses). The ML baseline produces no scores.}",
          r"\label{tab:scores}", r"\end{table}"]
    t += [r"\begin{table}[H]\centering\small", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lrrrrr}", r"\toprule",
          r" & \multicolumn{2}{c}{paper level (panel mean vs.\ reviewer mean)} & \multicolumn{3}{c}{reviewer level (matched bot vs.\ real reviewer)} \\",
          r"System & pooled $\rho$ & distance corr. & best match $\rho$ & random $\rho$ & worst match $\rho$ \\", r"\midrule"]
    for n, r in res.items():
        j = r["joint"]
        if not j:
            continue
        cell = lambda k: f"{f(j[k][0])} [{f(j[k][1])}, {f(j[k][2])}]"
        dc = f"{f(j['paper_dcor'][0])} ($p={j['paper_dcor'][1]:.3f}$)"
        t.append(f"{NAMES[n]} & {cell('paper_rho')} & {dc} & {cell('best_rho')} & {cell('rand_rho')} & {cell('worst_rho')} \\\\")
    t += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{Joint agreement of score vectors (overall, soundness, contribution; the rubric axes that both the real ICLR 2023 reviewers and our panels rate) on the 40 test papers. Pooled $\rho$: Spearman correlation after ranking each axis separately, computed over all (unit, axis) entries. Distance correlation: of the per-axis ranks, with a permutation $p$-value (1{,}000 shuffles of papers). Reviewer level: each real reviewer is matched to a bot by the Hungarian assignment on the $L_1$ distance of their score vectors (best: smallest total distance; worst: largest); random uses all bot--reviewer pairs. Brackets: bootstrap 95\% intervals over papers (500 resamples). The non-ML baseline's bots are real reviews of other papers.}",
          r"\label{tab:joint}", r"\end{table}"]
    for k in ("accepted", "rejected"):
        t += [r"\begin{table}[H]\centering\small", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
              r" & \multicolumn{4}{c}{Spearman $\rho$ (panel mean vs.\ reviewer mean)} & \multicolumn{3}{c}{mean signed error, panel $-$ real} \\",
              r"System & overall & soundness & contribution & joint & overall & soundness & contribution \\", r"\midrule"]
        for n, r in res.items():
            d = r["by_dec"].get(k)
            if not d:
                continue
            s_, b_, j = d["scores"], d["bias"], d["joint"]
            cell = lambda v: f"{f(v[0])} [{f(v[1])}, {f(v[2])}]"
            t.append(f"{NAMES[n]} & " + " & ".join(cell(s_[a]) for a in AXES) + f" & {cell(j['paper_rho']) if j else 'n/a'} & "
                     + " & ".join(cell(b_[a]) for a in AXES) + " \\\\")
        nk = next(r["by_dec"][k]["scores"]["n"] for r in res.values() if r["by_dec"].get(k))
        t += [r"\bottomrule", r"\end{tabular}}",
              rf"\caption{{Score agreement on the {nk} test papers that were finally {k}. $\rho$: Spearman correlation across papers of the panel's mean score and the real reviewers' mean score; joint: pooled within-axis $\rho$ over overall, soundness and contribution (as in Table~\ref{{tab:joint}}). Mean signed error: panel mean minus real mean on the $[0,1]$ rescaled rubric; a positive value means the panel is more optimistic than the real reviewers. Brackets: bootstrap 95\% intervals over papers. Within one decision group the range of real scores is narrow, so the correlations are weaker than on all papers.}}",
              rf"\label{{tab:scores_{k}}}", r"\end{table}"]
    (REP / "extra_tables.tex").write_text("\n".join(t))
    for n, r in res.items():
        for k, d in r["by_dec"].items():
            print(" ", k, n, {a: [round(x, 2) for x in d["scores"][a]] for a in AXES}, "bias", {a: [round(x, 2) for x in d["bias"][a]] for a in AXES})
    for n, r in res.items():
        s = r["scores"]
        if r["joint"]:
            print("  joint", n, {k: [None if x is None else round(x, 3) for x in v] for k, v in r["joint"].items() if k != "n"}, r["joint"]["n"])
        print(n, {k: round(v, 1) for k, v in r["conf"].items()}, "| scores:", None if not s else {a: [round(x, 2) for x in s[a]] for a in AXES} | {"auc": round(s["auc"], 2), "weak": s["weak_axis"]})


if __name__ == "__main__":
    main()
