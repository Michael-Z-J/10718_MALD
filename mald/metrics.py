"""evaluation suite. unit = one held-out paper; every metric is a per-paper value
that is then averaged over papers (paired bootstrap for system differences).

gt concerns come from mald.gt.clusters (llm-parsed items, minilm greedy grouping).
"same concern" test (issue metrics), primary: minilm cosine between a system
issue and any text of a gt concern >= tau. no single tau is picked: each
metric is averaged over the grid TAUS (area under the metric-vs-tau curve,
normalized), and the curve is kept. second matcher (a check only): llm judge.

meta-review level:
  recall_union     = share of gt concerns matched by any meta issue
  recall_multi     = same, only concerns raised by >= 2 real reviewers
  recall_w         = consensus-weighted recall
  precision        = share of meta issues that match some gt concern
  ndcg_cons/def    = ndcg of the ranked meta list, graded gain = consensus or
                     deficit relevance of the matched gt concern (each concern
                     credited once, at its first match)
  recall_at_k      = consensus-weighted recall of the top-k meta issues
  resolved_recall  = share of rebuttal-resolved gt concerns matched
  resolved_flagged = share of those whose matching meta issue is relevant and doable
  flag_lift        = P(match | relevant flag) - P(match | not relevant flag)
  p5_legacy        = precision@5 of evaluation/match.py (minilm, tau 0.5,
                     top 5, one-to-one)
panel level (same matcher, same tau grid):
  coverage(b, r)   = share of real reviewer r's items that bot b states
  alloc_best/worst = mean coverage over real reviewers after the hungarian
                     assignment that maximizes / minimizes total coverage
  alloc_rand       = expected coverage under a uniformly random assignment
  panel_union      = share of all real items stated by some bot
scores (paper level, across papers; no matcher involved):
  spearman / mae of panel-mean vs real-mean per axis, accept auc of the
  panel-mean overall, meta decision accuracy, score-allocated per-reviewer mae.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from mald import embed, llm

JUDGE = "glimmer"
TAUS = [round(0.40 + 0.05 * i, 2) for i in range(9)]
MATCH = """Below is one concern that a real reviewer raised about a paper, and a numbered list of issues that an automated reviewer raised about the same paper. Which listed issues state substantially the same concern: the same specific problem with the paper, not only the same general topic? Return their numbers; return an empty list if none does.

REAL REVIEWER CONCERN:
{concern}

AUTOMATED ISSUES:
{issues}"""
MATCH_SCHEMA = {"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "integer"}}}, "required": ["ids"]}
AXES = ["soundness", "contribution", "overall"]


def judge(concern_texts, issues):
    """ids (0-based) of issues that match a gt concern."""
    if not issues:
        return []
    lines = "\n".join(f"{i + 1}. {x['issue']}" for i, x in enumerate(issues))
    concern = "\n".join(f"- {t}" for t in concern_texts[:4])
    out, _ = llm.chat(JUDGE, [{"role": "user", "content": MATCH.format(concern=concern, issues=lines)}], MATCH_SCHEMA,
                      max_tokens=200, tag="match")
    return sorted({i - 1 for i in out.get("ids", []) if 1 <= i <= len(issues)})


def dcg(gains):
    return sum(g / np.log2(i + 2) for i, g in enumerate(gains))


def ndcg(rows, meta_issues, rel):
    first = {}
    for g, ids in enumerate(rows):
        for i in ids:
            first.setdefault(i, []).append(g)
    credited, gains = set(), []
    for i in range(len(meta_issues)):
        cand = [g for g in first.get(i, []) if g not in credited]
        best = max(cand, key=lambda g: rel[g]) if cand else None
        if best is not None:
            credited.add(best)
        gains.append(rel[best] if best is not None else 0.0)
    if not meta_issues:
        return 0.0  # an empty meta-review tells the author nothing
    ideal = dcg(sorted(rel, reverse=True)[: len(meta_issues)])
    return dcg(gains) / ideal if ideal > 0 else np.nan


def legacy_p5(gt, meta_issues, tau=0.5):
    """precision@5 of evaluation/match.py: one-to-one hungarian minilm matching at tau."""
    top = [x["issue"] for x in meta_issues[:5]]
    real = [it["text"] for it in gt["items"]]
    if not top or not real:
        return np.nan
    s = embed.vecs(top, "minilm") @ embed.vecs(real, "minilm").T
    r, c = linear_sum_assignment(-s)
    return float(sum(s[i, j] >= tau for i, j in zip(r, c)) / 5)


def sims(cl, issues):
    """minilm cosine, gt concern x meta issue (max over the concern's texts)."""
    if not cl or not issues:
        return np.zeros((len(cl), len(issues)))
    iv = embed.vecs([x["issue"] for x in issues], "minilm")
    return np.array([(embed.vecs(c["texts"], "minilm") @ iv.T).max(0) for c in cl])


def issue_metrics(cl, rows, issues):
    """per-paper counts as {metric: [numerator, denominator]}.

    ratio metrics pool over a population as sum(num) / sum(den); ndcg is a
    per-list value stored as [value, 1], so pooling gives the mean over lists.
    """
    hit = np.array([bool(r) for r in rows])
    cons = np.array([c["consensus"] for c in cl], float)
    defi = np.array([c["deficit"] for c in cl], float)
    multi = np.array([c["consensus"] >= 2 for c in cl])
    matched = set(i for r in rows for i in r)
    out = dict(recall_union=[hit.sum(), len(cl)], recall_multi=[(hit & multi).sum(), multi.sum()],
               recall_w=[(cons * hit).sum(), cons.sum()], precision=[len(matched), len(issues)],
               ndcg_cons=[ndcg(rows, issues, cons), 1], ndcg_def=[ndcg(rows, issues, defi), 1])
    for k in (1, 3, 5, 10, 15, 20):
        hk = np.array([any(i < k for i in r) for r in rows])
        out[f"recall_at_{k}"] = [(cons * hk).sum(), cons.sum()]
    res = [r for c, r in zip(cl, rows) if c["resolved"]]
    out["resolved_recall"] = [sum(bool(r) for r in res), len(res)]
    out["resolved_flagged"] = [sum(any(issues[i].get("relevant") and issues[i].get("doable") for i in r) for r in res if r),
                               sum(bool(r) for r in res)]
    rel = [x.get("relevant") for x in issues]
    m = [i in matched for i in range(len(issues))]
    out["match_if_relevant"] = [sum(a for a, r in zip(m, rel) if r is True), sum(r is True for r in rel)]
    out["match_if_irrelevant"] = [sum(a for a, r in zip(m, rel) if r is False), sum(r is False for r in rel)]
    return {k: [float(v[0]), float(v[1])] for k, v in out.items()}


def meta_level(gt, rec, cgrid, use_judge=False):
    """counts for every (grouping threshold c, matching threshold tau) cell.

    returns {"cells": {"c|tau": counts}, "judge": {"c": counts}, scalars}.
    the judge (a check only) runs at the middle grouping threshold.
    """
    from mald import gt as G
    issues = rec["meta"]["issues"]
    out = dict(cells={}, judge={}, n_issues=len(issues), p5_legacy=legacy_p5(gt, issues))
    for c in cgrid:
        cl = G.clusters(gt, c)
        if not cl:
            continue
        S = sims(cl, issues)
        for t in TAUS:
            out["cells"][f"{c}|{t}"] = issue_metrics(cl, [list(np.nonzero(S[g] >= t)[0]) for g in range(len(cl))], issues)
        if use_judge and c == cgrid[len(cgrid) // 2]:
            out["judge"][str(c)] = issue_metrics(cl, [judge(x["texts"], issues) for x in cl], issues)
    return out


def pool(papers, metric, cells=None, src="cells"):
    """population value: sum of numerators / sum of denominators over the
    papers and the chosen cells (all cells if None)."""
    num = den = 0.0
    for p in papers:
        for key, cnt in p[src].items():
            if (cells is None or key in cells) and metric in cnt and not np.isnan(cnt[metric][0]):
                num += cnt[metric][0]
                den += cnt[metric][1]
    return num / den if den else float("nan")


def coverage(gt, bots, tau):
    """coverage matrix (bots x real reviewers) and union coverage at tau."""
    names = [r["name"] for r in gt["reviewers"]]
    cov = np.zeros((len(bots), len(names)))
    if not gt["items"]:
        return cov, np.nan
    allv = embed.vecs([it["text"] for it in gt["items"]], "minilm")
    hit_any = np.zeros(len(gt["items"]), bool)
    for b, bot in enumerate(bots):
        ws = [w["issue"] for w in bot["weaknesses"]]
        if not ws:
            continue
        s = (embed.vecs(ws, "minilm") @ allv.T).max(0) >= tau
        hit_any |= s
        for j, n in enumerate(names):
            idx = [k for k, it in enumerate(gt["items"]) if it["reviewer"] == n]
            cov[b, j] = s[idx].mean() if idx else np.nan
    return cov, float(hit_any.mean())


def assign(cost):
    """hungarian on a bots x reviewers cost; replicate bots if reviewers outnumber them."""
    nb, nr = cost.shape
    rep = int(np.ceil(nr / nb))
    c = np.tile(cost, (rep, 1))
    r, k = linear_sum_assignment(c)
    return [(i % nb, j) for i, j in zip(r, k)]


def panel_level(gt, rec):
    bots = rec["panel"]
    if not bots or not gt["reviewers"]:
        return {}
    acc = []
    for t in TAUS:
        cov, union = coverage(gt, bots, t)
        cov = np.nan_to_num(cov)
        acc.append(dict(alloc_best=float(np.mean([cov[i, j] for i, j in assign(-cov)])),
                        alloc_worst=float(np.mean([cov[i, j] for i, j in assign(cov)])),
                        alloc_rand=float(cov.mean()), panel_union=union))
    out = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
    # score allocation on available axes
    real = [[r["scores"].get(a) for a in AXES] for r in gt["reviewers"]]
    bot = [[b["scores"].get(a) for a in AXES] for b in bots]
    if all(x is not None for row in bot for x in row):
        d = np.array([[np.nanmean([abs(bb - rr) for bb, rr in zip(bv, rv) if rr is not None]) for rv in real] for bv in bot])
        out["score_mae_best"] = float(np.mean([d[i, j] for i, j in assign(d)]))
        out["score_mae_worst"] = float(np.mean([d[i, j] for i, j in assign(-d)]))
    return out


def paper_scores(gt, rec, accepted):
    real = {a: np.nanmean([r["scores"][a] if r["scores"][a] is not None else np.nan for r in gt["reviewers"]]) for a in AXES}
    pm = rec["meta"].get("panel_mean") or ({a: np.mean([b["scores"][a] for b in rec["panel"]]) for a in AXES}
                                         if rec["panel"] and all(b["scores"].get(a) is not None for b in rec["panel"] for a in AXES) else None)
    ms = rec["meta"].get("scores") or {}
    return dict(real=real, panel=pm, meta={a: ms.get(a) for a in AXES}, accepted=accepted,
                decision=rec["meta"].get("decision"),
                real_sd=float(np.std([r["scores"]["overall"] for r in gt["reviewers"]])),
                panel_sd=float(np.std([b["scores"]["overall"] for b in rec["panel"]])) if rec["panel"] else None)


def score_summary(rows):
    """across-paper score agreement for one system."""
    out = {}
    for src in ("panel", "meta"):
        for a in AXES:
            pairs = [(r[src][a], r["real"][a]) for r in rows if r[src] and r[src].get(a) is not None and not np.isnan(r["real"][a])]
            if len(pairs) >= 5:
                x, y = map(np.array, zip(*pairs))
                out[f"{src}_{a}_spearman"] = float(spearmanr(x, y).statistic)
                out[f"{src}_{a}_mae"] = float(np.abs(x - y).mean())
        acc = [(r[src]["overall"], r["accepted"]) for r in rows if r[src] and r[src].get("overall") is not None]
        if len(acc) >= 5 and len({a for _, a in acc}) == 2:
            out[f"{src}_accept_auc"] = float(roc_auc_score([a for _, a in acc], [s for s, _ in acc]))
    dec = [(r["decision"] == "accept", r["accepted"]) for r in rows if r["decision"]]
    if dec:
        out["decision_acc"] = float(np.mean([p == a for p, a in dec]))
    sd = [(r["panel_sd"], r["real_sd"]) for r in rows if r["panel_sd"] is not None]
    if len(sd) >= 5:
        out["disagreement_spearman"] = float(spearmanr(*zip(*sd)).statistic)
    return out
