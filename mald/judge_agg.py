"""deterministic aggregation of the llm-judge labels (secondary metrics).

judge/out/<forum>.json + judge/key.json + system records + gt
  --> per system, pooled over papers (sum of numerators / sum of denominators):
  validity       share of meta issues judged valid ("yes"); valid_any adds "partial"
  valid_at5      same, top-5 issues only
  imp_ndcg       ndcg of the system's own order, gain = judged importance of valid
                 issues (yes: importance, partial: importance / 2, no: 0)
  hi_valid       valid issues of importance 3 per paper
  item_recall    share of real reviewer items stated by some issue (judge matches)
  rev_recall     macro share over real reviewers of their items covered
  shared         issues that state items of >= 2 different real reviewers, per paper
  precision      share of issues that match at least one real item
  recall_at_k    share of real items stated by the top-k issues
  flag confusion system doable flag vs judged doable, on valid issues (2x2, kappa);
                 system relevant flag vs judged (valid yes and importance >= 2)
  alloc          judged coverage(bot, real reviewer) --> hungarian best / worst,
                 mean over pairs (random allocation)
  rebut          real items labelled resolved / open (reviewer follow-ups) x
                 (missed / found and flagged doable / found and not flagged)
  curve_c        the issue metrics of mald.metrics with the judge as matcher, for each
                 gt grouping threshold c: a gt concern is found if the judge matched
                 one of its items; the same 12 papers with the minilm matcher (mean
                 over tau) are kept for comparison
paired bootstrap over papers (seed 0, 2000 resamples) for differences to b1.

usage: MALD_JUDGE=<dir>[:<dir> ...] python -m mald.judge_agg > report/judge_tables.tex
systems are read from each dir's key.json (any number; one judge run per dir);
differences are to b1 when present. agg.json is written to the first dir.
"""
import json, sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from mald import gt as G, jl, metrics as M
from mald.evaluate import CGRID
from mald.paths import JUDGE, OUT

JS = [Path(d) for d in str(JUDGE).split(":")]
J = JS[0]
LABEL = {"m": "Ours (agentic)", "b1": "Ours", "b1old": "ML baseline", "b0": "Non-ML baseline",
         "m_unc": "Ours (agentic), no rebuttal-mode", "m_alt": "Ours (agentic), LLM meta-review",
         "m_nemotron": "Ours (agentic), Nemotron", "b1_nemotron": "Ours, Nemotron", "m_mix2": "Ours (agentic), mixed panel",
         "m_qwen": "Ours (agentic), Qwen", "b1_qwen": "Ours, Qwen"}
VAL = {"yes": 1.0, "partial": 0.5, "no": 0.0}


def dcg(g):
    return sum(x / np.log2(i + 2) for i, x in enumerate(g))


def flagged(x):
    return bool(x.get("doable")) and x.get("relevant") is not False


def paper(fid, sysname, lab, gt, rec):
    """per-paper counts for one system: {metric: [num, den]} plus 2x2 tables."""
    iss = lab["issues"]
    meta = rec["meta"]["issues"]
    items = gt["items"]
    rev = {}
    for i, it in enumerate(items):
        rev.setdefault(it["reviewer"], []).append(i)
    c = {}
    n = len(iss)
    yes = [x["valid"] == "yes" for x in iss]
    c["validity"] = [sum(yes), n]
    c["valid_any"] = [sum(x["valid"] != "no" for x in iss), n]
    c["valid_at5"] = [sum(yes[:5]), min(n, 5)]
    gain = [VAL[x["valid"]] * x["importance"] for x in iss]
    ideal = dcg(sorted(gain, reverse=True))
    c["imp_ndcg"] = [dcg(gain) / ideal if ideal > 0 else 0.0, 1]
    c["hi_valid"] = [sum(x["valid"] == "yes" and x["importance"] == 3 for x in iss), 1]
    hit = set(j for x in iss for j in x["matches"] if 0 <= j < len(items))
    c["item_recall"] = [len(hit), len(items)]
    c["rev_recall"] = [sum(len(hit & set(v)) / len(v) for v in rev.values()), len(rev)]
    c["shared"] = [sum(len({items[j]["reviewer"] for j in x["matches"] if 0 <= j < len(items)}) >= 2 for x in iss), 1]
    c["precision"] = [sum(bool(x["matches"]) for x in iss), n]
    for k in (1, 3, 5, 10):
        hk = set(j for x in iss[:k] for j in x["matches"] if 0 <= j < len(items))
        c[f"recall_at_{k}"] = [len(hk), len(items)]
    # flag confusion on valid issues: rows judge (T/F), cols system flag (T/F)
    fd = np.zeros((2, 2))
    fr = np.zeros((2, 2))
    for x, m in zip(iss, meta):
        if x["valid"] == "no":
            continue
        if m.get("doable") is not None:
            fd[int(not x["doable"]), int(not m["doable"])] += 1
        if m.get("relevant") is not None:
            fr[int(not (x["valid"] == "yes" and x["importance"] >= 2)), int(not m["relevant"])] += 1
    # rebuttable confusion on real items
    reb = {k: dict(miss=0, flag=0, noflag=0) for k in ("resolved", "open")}
    by_item = {}
    for x, m in zip(iss, meta):
        for j in x["matches"]:
            by_item.setdefault(j, []).append(m)
    for j, it in enumerate(items):
        if it["status"] not in reb:
            continue
        ms = by_item.get(j, [])
        col = "miss" if not ms else "flag" if any(flagged(m) for m in ms) else "noflag"
        reb[it["status"]][col] += 1
    # reviewer allocation from judged bot coverage
    names = sorted(rev)
    bots = lab["bots"]
    alloc = {}
    if bots and names:
        cov = np.array([[len(set(b["covers"]) & set(rev[r])) / len(rev[r]) for r in names] for b in bots])
        rep = int(np.ceil(len(names) / len(bots)))
        big = np.tile(cov, (rep, 1))
        rb_, cb_ = linear_sum_assignment(-big)
        rw_, cw_ = linear_sum_assignment(big)
        alloc = dict(best=float(big[rb_, cb_].mean()), worst=float(big[rw_, cw_].mean()), rand=float(cov.mean()))
    return c, fd, fr, reb, alloc


def curve(lab, gt, rec):
    """judge-matched issue metric counts per gt grouping threshold c."""
    iss = lab["issues"]
    meta = rec["meta"]["issues"]
    out = {}
    for c in CGRID:
        cl = G.clusters(gt, c)
        if not cl:
            continue
        rows = [sorted(i for i, x in enumerate(iss) if set(x["matches"]) & set(k["items"])) for k in cl]
        out[str(c)] = M.issue_metrics(cl, rows, meta)
    return out


def emb_curve(name, fids):
    """minilm-matched pooled metrics per c on the given papers (mean over tau)."""
    rs = [r for r in json.loads((OUT / "test" / f"eval_{name}.json").read_text()) if r["forum"] in fids]
    return {m: [M.pool(rs, m, {f"{c}|{t}" for t in M.TAUS}) for c in CGRID] for m in CURVED}


CURVED = ("recall_w", "recall_union", "recall_multi", "ndcg_cons", "precision")


def kappa(t):
    n = t.sum()
    if n == 0:
        return float("nan")
    po = np.trace(t) / n
    pe = (t.sum(0) * t.sum(1)).sum() / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def pool(per, m):
    num = sum(p[0][m][0] for p in per)
    den = sum(p[0][m][1] for p in per)
    return num / den if den else float("nan")


def boot(a, b, m, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    an = np.array([p[0][m][0] for p in a]); ad = np.array([p[0][m][1] for p in a])
    bn = np.array([p[0][m][0] for p in b]); bd = np.array([p[0][m][1] for p in b])
    idx = rng.integers(0, len(a), (n, len(a)))
    with np.errstate(invalid="ignore", divide="ignore"):
        ds = an[idx].sum(1) / ad[idx].sum(1) - bn[idx].sum(1) / bd[idx].sum(1)
    d0 = an.sum() / ad.sum() - bn.sum() / bd.sum()
    return [float(d0), float(np.nanquantile(ds, 0.025)), float(np.nanquantile(ds, 0.975))]


def main():
    gts = {r["forum"]: r for r in jl.rows(OUT / "gt_test.jsonl")}
    order = list(LABEL)
    keys = [(d, json.loads((d / "key.json").read_text())) for d in JS]
    systems = sorted({s for _, k in keys for v in k.values() for s in v.values()}, key=lambda s: (order.index(s) if s in order else len(order), s))
    recs = {s: {r["forum"]: r for r in jl.rows(OUT / "test" / f"{s}.jsonl")} for s in systems}
    per = {s: [] for s in systems}
    curves = {s: [] for s in systems}
    judged_items = []
    done = set()
    for d, key in keys:
        for fid in sorted(f for f in key if (d / "out" / f"{f}.json").exists()):
            lab = json.loads((d / "out" / f"{fid}.json").read_text())
            for letter, s in sorted(key[fid].items()):
                per[s].append(paper(fid, s, lab["systems"][letter], gts[fid], recs[s][fid]))
                curves[s].append(dict(forum=fid, cells=curve(lab["systems"][letter], gts[fid], recs[s][fid])))
            if d == JS[0]:
                st = {i: it["status"] for i, it in enumerate(gts[fid]["items"])}
                judged_items += [(st[x["id"]], x["doable"]) for x in lab["real_items"] if x["id"] in st]
            done.add(fid)
    done = sorted(done)
    res = {}
    for s, p in per.items():
        r = {m: pool(p, m) for m in p[0][0]} if p else {}
        fd = sum(x[1] for x in p); fr = sum(x[2] for x in p)
        r["doable_flag"] = dict(table=fd.tolist(), acc=float(np.trace(fd) / fd.sum()) if fd.sum() else float("nan"), kappa=kappa(fd))
        r["relevant_flag"] = dict(table=fr.tolist(), acc=float(np.trace(fr) / fr.sum()) if fr.sum() else float("nan"), kappa=kappa(fr))
        r["rebut"] = {k: {c: sum(x[3][k][c] for x in p) for c in ("miss", "flag", "noflag")} for k in ("resolved", "open")}
        al = [x[4] for x in p if x[4]]
        r["alloc"] = {k: float(np.mean([a[k] for a in al])) for k in ("best", "worst", "rand")} if al else {}
        r["n_papers"] = len(p)
        fids = {c["forum"] for c in curves[s]}
        r["curve_c"] = {m: [M.pool(curves[s], m, {str(c)}) for c in CGRID] for m in CURVED}
        r["emb_curve_c"] = emb_curve(s, fids)
        if s != "b1" and per.get("b1"):
            r["boot_vs_b1"] = {m: boot(p, per["b1"], m) for m in ("validity", "imp_ndcg", "item_recall", "precision", "rev_recall")}
        res[s] = r
    jt = {k: [sum(d for st, d in judged_items if st == k), sum(st == k for st, _ in judged_items)] for k in ("resolved", "open")}
    res["_judge_doable_vs_rebuttal"] = jt
    res["_papers"] = done
    res["_systems"] = systems
    res["_cgrid"] = CGRID
    (J / "agg.json").write_text(json.dumps(res, indent=1, default=float))
    tex(res)


def tex(res):
    f = lambda x: "n/a" if x is None or x != x else f"{x:.2f}"
    S = [s for s in res["_systems"] if res.get(s)]
    n = len(res["_papers"])
    head = ["valid", "valid$_{\\pm}$", "valid@5", "nDCG$_{imp}$", "hi/paper", "item rec", "rev rec", "shared/paper", "prec", "rec@5", "rec@10"]
    keys = ["validity", "valid_any", "valid_at5", "imp_ndcg", "hi_valid", "item_recall", "rev_recall", "shared", "precision", "recall_at_5", "recall_at_10"]
    out = [r"\begin{table}[H]\centering\small", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l" + "r" * len(keys) + "}", r"\toprule",
           "system & " + " & ".join(head) + r" \\", r"\midrule"]
    out += [f"{LABEL.get(s, s)} & " + " & ".join(f(res[s].get(k)) for k in keys) + r" \\" for s in S]
    out += [r"\bottomrule", r"\end{tabular}}",
            rf"\caption{{LLM-judge (Claude Opus 5.5 subagents, blinded to system identity) on {n} test papers. valid: share of meta-review issues that the paper really has; valid$_\pm$ adds partially valid; nDCG$_{{imp}}$: the system's own order scored with judged importance; hi/paper: valid issues of importance 3 per paper; item/rev rec: real reviewer items (pooled / macro over reviewers) stated by some issue, decided by the judge (no similarity threshold); shared/paper: issues that state items of two or more real reviewers; prec: issues matching any real item. Pooled population values. The ML baseline was judged in a separate run with the same specification, without the other systems beside it.}}",
            r"\end{table}"]
    out += [r"\begin{table}[H]\centering\small", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
            r"system & doable acc & doable $\kappa$ & relevant acc & relevant $\kappa$ & res: miss/flag/no & open: miss/flag/no & alloc best/rand/worst \\", r"\midrule"]
    for s in S:
        r = res[s]
        rb = r["rebut"]
        al = r["alloc"]
        out.append(f"{LABEL.get(s, s)} & {f(r['doable_flag']['acc'])} & {f(r['doable_flag']['kappa'])} & {f(r['relevant_flag']['acc'])} & {f(r['relevant_flag']['kappa'])} & "
                   f"{rb['resolved']['miss']}/{rb['resolved']['flag']}/{rb['resolved']['noflag']} & {rb['open']['miss']}/{rb['open']['flag']}/{rb['open']['noflag']} & "
                   f"{f(al.get('best'))}/{f(al.get('rand'))}/{f(al.get('worst'))} \\\\")
    jt = res["_judge_doable_vs_rebuttal"]
    out += [r"\bottomrule", r"\end{tabular}}",
            rf"\caption{{LLM-judge flag and allocation metrics ({n} test papers). Flag agreement is computed on issues the judge found valid: the system's own doable / relevant flags vs.\ the judge's labels (accuracy and Cohen's $\kappa$). res / open: real reviewer items that the reviewer later called resolved / still open, split into missed, found and flagged doable, found and not flagged (counts). alloc: mean judged coverage of a real reviewer's items by its allocated bot under the best / random / worst assignment. The ML baseline sets no flags and has no panel, so its flag and allocation cells are n/a. For reference, the judge itself labels {jt['resolved'][0]}/{jt['resolved'][1]} resolved and {jt['open'][0]}/{jt['open'][1]} still-open real items as doable.}}",
            r"\end{table}"]
    print("\n".join(out))


if __name__ == "__main__":
    main()
