"""the systems under comparison. each maps one paper to a standard record.

record = {forum, system, panel: [bot], meta: {issues, scores, decision}, calls}
bot    = {mode, weaknesses: [{issue, axis, fix, effort}], scores (in [0, 1])}

b0     non-ml baseline: nearest pool papers --> one real review of each as a bot
       --> rule-based weakness sentences --> category vote as the meta-review.
b1old  ml baseline: one prompt for the top-5 weaknesses (no panel, no scores).
b1     ours: one prompt writes six reviews and the meta-review.
m      ours, agentic: six reviewers, each given a real review of a related
       paper (unconditional or rebuttal mode) --> clustering of their issues
       --> ranked issue table, optionally rewritten by a meta-review call.
"""
import random, re
from functools import cache

import numpy as np
import polars as pl

from baseline import nonMLBaseline as nml
from mald import data, embed, llm, prompts as P

AX = P.AXES + ["overall"]
CAT_AXIS = dict(baselines="soundness", experiments="soundness", soundness="soundness", analysis="soundness",
                clarity="presentation", reproducibility="presentation", novelty="contribution",
                motivation="contribution", related_work="contribution", efficiency="contribution")
ACCEPT = (6 - 1) / 9
MAXTOK = 24000


def norm(s):
    """form scores --> [0, 1] (iclr 2022/2023 scale, as in the gt scaling)."""
    return {a: (s[a] - 1) / 3 for a in P.AXES} | {"overall": (s["overall"] - 1) / 9}


def paper(forum, model="gemma"):
    return llm.clip(model, data.text(forum), MAXTOK)


# retrieval

@cache
def pool_index(year):
    """pool forums for a target year with bge vectors of title + abstract."""
    f = data.pool(year)
    v = embed.vecs([f"{t}. {a}" for t, a in zip(f["title"], f["abstract"])])
    return f, v


def neighbors(forum, year, k):
    f, v = pool_index(year)
    me = data.forums().filter(pl.col("forum") == forum).row(0, named=True)
    q = embed.vecs([f"{me['title']}. {me['abstract']}"])[0]
    idx = np.argsort(-(v @ q))[:k]
    return f[idx.tolist()]


ENGAGED = re.compile(
    r"(rais|increas|updat|bump|improv|adjust)\w*\s+(my\s+|the\s+)?(score|rating)|"
    r"(concerns?|questions?|issues?)\s+(have\s+been|has\s+been|are|were|was|is)\s+(\w+\s+)?(addressed|resolved|answered|clarified)",
    re.I)


def followups():
    """reviewer follow-ups that say the reviewer changed their mind, by (forum, reviewer sig)."""
    c = data.comments().filter((pl.col("role") == "Reviewer") & pl.col("text").str.contains(ENGAGED.pattern))
    return {(r["forum"], r["sig"]): r["text"] for r in c.iter_rows(named=True)}


@cache
def _followups():
    return followups()


def exemplars(forum, year, n_unc, n_reb, seed):
    """pick the real reviews that the panel reviewers are given.

    unconditional: the reviews of the 40 nearest pool papers, split into
      overall-score terciles; reviewer i gets one of the 3 nearest reviews
      of tercile i mod 3, so the panel spans lenient to strict reviewers.
    rebuttal: the nearest reviews whose author wrote a follow-up that says
      the authors' response changed their mind.
    """
    nb = neighbors(forum, year, 200)
    rank = {f: i for i, f in enumerate(nb["forum"])}
    rv = data.reviews().filter(pl.col("forum").is_in(nb["forum"].to_list())).with_columns(
        pl.col("forum").replace_strict(rank).alias("rank")).sort("rank")
    near = rv.filter(pl.col("rank") < 40)
    q = np.quantile(near["overall"].to_numpy(), [1 / 3, 2 / 3])
    terc = [near.filter(pl.col("overall") <= q[0]), near.filter((pl.col("overall") > q[0]) & (pl.col("overall") <= q[1])),
            near.filter(pl.col("overall") > q[1])]
    rng = random.Random(seed)
    unc, used = [], set()
    for i in range(n_unc):
        t = terc[i % 3] if len(terc[i % 3]) else near
        rows = [r for r in t.iter_rows(named=True) if r["forum"] not in used][:3]
        r = rng.choice(rows)
        used.add(r["forum"])
        unc.append(r)
    fu = _followups()
    reb = []
    for r in rv.iter_rows(named=True):
        if len(reb) == n_reb:
            break
        if (r["forum"], r["sig"]) in fu and r["forum"] not in used:
            used.add(r["forum"])
            reb.append(r | {"followup": fu[(r["forum"], r["sig"])]})
    return unc, reb


def form_overall(x):
    return int(round(1 + 9 * x))


# b0

def b0(forum, year, seed=0):
    nb = neighbors(forum, year, 6)
    me = data.forums().filter(pl.col("forum") == forum).row(0, named=True)
    rng = random.Random(seed)
    panel, cats = [], {}
    for f in nb["forum"]:
        rs = data.reviews().filter(pl.col("forum") == f).rows(named=True)
        r = rng.choice(rs)
        sents = nml.weakness_sentences(r["text"])
        ws = []
        for s in sents:
            c = nml.categorize(s)
            if c:
                ws.append(dict(issue=s, axis=CAT_AXIS[c[0]], fix=nml.ACTIONS[c[0]], effort="weeks"))
                for k in c:
                    cats.setdefault(k, []).append((len(panel), s))
        panel.append(dict(mode="neighbor", weaknesses=ws, scores={a: r[a] for a in AX}))
    from sklearn.feature_extraction.text import TfidfVectorizer
    q = f"{me['title']}. {me['abstract']}"
    issues = []
    order = sorted(cats, key=lambda c: (-len({b for b, _ in cats[c]}), list(CAT_AXIS).index(c)))
    taken = set()
    for c in order:
        sents = [s for _, s in cats[c] if s not in taken] or [s for _, s in cats[c]]
        vec = TfidfVectorizer().fit(sents + [q])
        best = sents[int(np.argmax((vec.transform(sents) @ vec.transform([q]).T).toarray()[:, 0]))]
        taken.add(best)
        bots = sorted({b + 1 for b, _ in cats[c]})
        issues.append(dict(issue=best, axis=CAT_AXIS[c], raised_by=bots, relevant=len(bots) >= 2, doable=c != "novelty"))
    sc = {a: float(np.nanmean([b["scores"][a] if b["scores"][a] is not None else np.nan for b in panel]))
          if any(b["scores"][a] is not None for b in panel) else None for a in AX}
    return dict(panel=panel, meta=dict(issues=issues, scores=sc, decision="accept" if sc["overall"] >= ACCEPT else "reject"),
                calls=[])


# b1 / b1old

def b1(forum, year, model="gemma", seed=0, temp=0.7):
    msg = [{"role": "system", "content": P.HEADER}, {"role": "user", "content": P.B1.format(paper=paper(forum, model))}]
    out, u = llm.chat(model, msg, P.PANEL, temp=temp, seed=seed, max_tokens=14000, tag="b1")
    if "_error" in out:
        return dict(panel=[], meta=dict(issues=[], scores={a: None for a in AX}, decision=None), calls=[u], error=True)
    panel = [dict(mode="b1", weaknesses=r["weaknesses"], scores=norm(r["scores"])) for r in out["reviews"]]
    m = out["meta"]
    return dict(panel=panel, meta=dict(issues=m["issues"], scores=norm(m["scores"]), decision=m["decision"],
                                       summary=m["summary"]), calls=[u])


def b1old(forum, year, model="gemma", seed=0):
    msg = [{"role": "user", "content": P.B1_OLD.format(paper=paper(forum, model))}]
    out, u = llm.chat(model, msg, P.B1_OLD_SCHEMA, temp=0.0, seed=seed, max_tokens=3000, tag="b1old")
    ws = sorted(out.get("weaknesses", []), key=lambda w: -w["severity"])
    issues = [dict(issue=w["weakness"], axis=None, raised_by=[], relevant=None, doable=None) for w in ws]
    return dict(panel=[], meta=dict(issues=issues, scores={a: None for a in AX}, decision=None), calls=[u])


# m

def clip_words(t, n):
    w = t.split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def persona(kind, r):
    if kind == "none":
        return P.PERSONA_NONE
    if kind == "unc":
        return P.PERSONA_UNCOND.format(overall=form_overall(r["overall"]), review=clip_words(r["text"], 900))
    return P.PERSONA_REBUTTAL.format(review=clip_words(r["text"], 700), followup=clip_words(r["followup"], 300))


def panel(forum, year, model="gemma", n_reb=3, retrieval=True, seed=0, temp=0.7, models=None):
    """six reviewers. retrieval off --> the generic persona for all six.

    models: optional list of six generator names, one per reviewer (mixed
    panel). default: all reviewers use `model`.
    """
    models = models or [model] * 6
    if retrieval:
        unc, reb = exemplars(forum, year, 6 - n_reb, n_reb, seed)
        specs = [("unc", r) for r in unc] + [("reb", r) for r in reb]
        specs += [("unc", unc[i % len(unc)]) for i in range(6 - len(specs))]
    else:
        specs = [("none", None)] * 6
    txt = {mm: paper(forum, mm) for mm in set(models)}
    bots, calls = [], []
    from concurrent.futures import ThreadPoolExecutor

    def one(i):
        kind, r = specs[i]
        msg = [{"role": "system", "content": P.HEADER},
               {"role": "user", "content": P.REVIEWER.format(i=i + 1, paper=txt[models[i]], persona=persona(kind, r))}]
        return llm.chat(models[i], msg, P.REVIEW, temp=temp, seed=seed * 100 + i, max_tokens=3000, tag=f"m-{kind}")

    with ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(one, range(6)))
    for (kind, r), (out, u), mm in zip(specs, outs, models):
        calls.append(u)
        if "_error" in out:
            continue
        bots.append(dict(mode=kind, weaknesses=out["weaknesses"], scores=norm(out["scores"]),
                         exemplar=r["id"] if r else None, model=mm))
    return bots, calls


def aggregate(bots, theta=0.35, w=0.5):
    """issue table of the panel (no llm).

    weaknesses --> bge vectors --> average-linkage clustering, cosine distance
    cut theta --> per cluster: reviewers, modes, axis, effort votes,
    deficit = mean over raisers of (1 - raiser's score on the axis)
    --> priority = n_reviewers / 6 + w * deficit, sorted descending.
    """
    from sklearn.cluster import AgglomerativeClustering
    items = [(b, x) for b, bot in enumerate(bots) for x in bot["weaknesses"]]
    if not items:
        return []
    v = embed.vecs([x["issue"] for _, x in items])
    lab = [0] if len(items) == 1 else AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average", distance_threshold=theta).fit(v).labels_
    out = []
    for c in sorted(set(lab)):
        idx = [i for i, l in enumerate(lab) if l == c]
        cen = v[idx].mean(0)
        rep = sorted(idx, key=lambda i: -float(v[i] @ cen))
        raisers = sorted({items[i][0] for i in idx})
        axes = [items[i][1]["axis"] for i in idx]
        axis = max(set(axes), key=axes.count)
        defi = float(np.mean([1 - bots[b]["scores"][axis] for b in raisers]))
        eff = [items[i][1]["effort"] for i in idx]
        out.append(dict(texts=[items[i][1]["issue"] for i in rep[:3]], fix=items[rep[0]][1]["fix"],
                        bots=[b + 1 for b in raisers], modes=sorted({bots[b]["mode"] for b in raisers}),
                        axis=axis, effort={e: eff.count(e) for e in set(eff)}, deficit=defi,
                        priority=len(raisers) / 6 + w * defi))
    return sorted(out, key=lambda c: -c["priority"])


def table(clusters, bots, show_modes, show_deficit):
    rows = []
    for i, c in enumerate(clusters, 1):
        who = ", ".join(f"R{b}" + (f" ({bots[b - 1]['mode']})" if show_modes else "") for b in c["bots"])
        extra = f"; score gap {c['deficit']:.2f}" if show_deficit else ""
        st = " | ".join(c["texts"])
        rows.append(f"{i}. [{len(c['bots'])} reviewers: {who}; axis {c['axis']}; effort {c['effort']}{extra}] {st} FIX: {c['fix']}")
    return "\n".join(rows)


def meta(forum, bots, clusters, model="gemma", modes=True, rubric=True, seed=0):
    me = data.forums().filter(pl.col("forum") == forum).row(0, named=True)
    mean = {a: float(np.mean([b["scores"][a] for b in bots])) for a in AX}
    sc = ""
    if rubric:
        form = {a: round(1 + 3 * mean[a], 2) for a in P.AXES} | {"overall": round(1 + 9 * mean["overall"], 2)}
        sc = f"\nMEAN REVIEWER SCORES: {form}. A low score on an axis means issues on that axis matter more for the decision.\n"
    note = (" Reviewers marked (reb) are reviewers who, on past papers, changed their minds after the authors fixed"
            " concerns; their issues are good starting points for revision.") if modes else ""
    msg = [{"role": "system", "content": P.HEADER},
           {"role": "user", "content": P.META_PROMPT.format(
               modes=note, title=me["title"], abstract=me["abstract"],
               deficit=", then score gap" if rubric else "", table=table(clusters, bots, modes, rubric), scores=sc)}]
    out, u = llm.chat(model, msg, P.META, temp=0.0, seed=seed, max_tokens=5000, tag="m-meta")
    if "_error" in out:
        return None, u
    return dict(issues=out["issues"], scores=norm(out["scores"]), decision=out["decision"], summary=out["summary"],
                panel_mean=mean), u


def det_meta(bots, clusters):
    """meta-review without an llm call: the ranked issue table."""
    mean = {a: float(np.mean([b["scores"][a] for b in bots])) for a in AX}
    issues = [dict(issue=c["texts"][0], axis=c["axis"], raised_by=c["bots"], relevant=len(c["bots"]) >= 2,
                   doable=c["effort"].get("months", 0) * 2 < sum(c["effort"].values())) for c in clusters]
    return dict(issues=issues, scores=mean, decision="accept" if mean["overall"] >= ACCEPT else "reject")


def m(forum, year, model="gemma", n_reb=3, retrieval=True, theta=0.35, w=0.5, modes=True, rubric=True,
      llm_meta=True, seed=0, models=None):
    bots, calls = panel(forum, year, model, n_reb, retrieval, seed, models=models)
    cl = aggregate(bots, theta, w)
    if llm_meta:
        mt, u = meta(forum, bots, cl, model, modes and retrieval, rubric, seed)
        calls.append(u)
        if mt is None:
            mt = det_meta(bots, cl) | {"meta_error": True}
    else:
        mt = det_meta(bots, cl)
    mt["panel_mean"] = {a: float(np.mean([b["scores"][a] for b in bots])) for a in AX}
    return dict(panel=bots, meta=mt, clusters=cl, calls=calls)
