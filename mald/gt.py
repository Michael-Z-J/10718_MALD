"""ground truth from real reviews.

parsing (judge llm, not the generator; run once, same for every system):
  real review --> atomic weaknesses (axis-tagged)
  reviewer follow-up comments --> which of that reviewer's weaknesses they call resolved
grouping (no llm, at eval time; the rule of evaluation/extract_weaknesses.dedup):
  paper's weaknesses --> greedy minilm clustering at cosine >= thr, at most one
  item per reviewer per cluster --> concerns with the reviewers who raised them

graded relevance of a cluster c (no free constants):
  consensus(c) = number of reviewers who raised c
  deficit(c)   = sum over raisers r of (1 - s_r,axis(c)), s in [0, 1]
                 (axis without a gt score --> the raiser's overall score)

usage: python -m mald.gt <split> [n]   (llm calls; cached)
"""
import json, os
from concurrent.futures import ThreadPoolExecutor

import polars as pl

from mald import data, llm

JUDGE = "glimmer"
AXES = ["soundness", "presentation", "contribution"]

EXTRACT = """Below is one peer review of an ML paper. List every distinct weakness or criticism that the reviewer raises about the paper, including the ones phrased as questions or requests. Rules:
- one weakness per item, one sentence, keep the specifics (method names, datasets, sections, tables);
- leave out praise and summaries of the paper;
- axis: "soundness" (correctness, evidence, experiments, baselines, theory), "presentation" (clarity, writing, missing details), "contribution" (novelty, significance, related work).

REVIEW:
{review}"""
EXTRACT_SCHEMA = {"type": "object", "properties": {"weaknesses": {"type": "array", "items": {
    "type": "object", "properties": {"text": {"type": "string"}, "axis": {"enum": AXES}}, "required": ["text", "axis"]}}},
    "required": ["weaknesses"]}

RESOLVE = """A reviewer raised the weaknesses below about a paper. After the authors' response, the reviewer wrote the follow-up comments below. For each weakness id, decide from the follow-up comments only: "resolved" if the reviewer says it was addressed or no longer a concern, "open" if the reviewer says it remains, "unknown" if the follow-up does not say.

WEAKNESSES:
{items}

FOLLOW-UP COMMENTS:
{follow}"""
RESOLVE_SCHEMA = {"type": "object", "properties": {"labels": {"type": "array", "items": {
    "type": "object", "properties": {"id": {"type": "integer"}, "status": {"enum": ["resolved", "open", "unknown"]}},
    "required": ["id", "status"]}}}, "required": ["labels"]}


def extract(review):
    out, _ = llm.chat(JUDGE, [{"role": "user", "content": EXTRACT.format(review=review["text"])}], EXTRACT_SCHEMA,
                      max_tokens=3000, tag="gt-extract")
    return out.get("weaknesses", [])


def resolve(items, follow):
    lines = "\n".join(f"{i}. {it['text']}" for i, it in enumerate(items))
    out, _ = llm.chat(JUDGE, [{"role": "user", "content": RESOLVE.format(items=lines, follow=follow)}], RESOLVE_SCHEMA,
                      max_tokens=2000, tag="gt-resolve")
    lab = {d["id"]: d["status"] for d in out.get("labels", [])}
    return [lab.get(i, "unknown") for i in range(len(items))]


def paper(forum):
    """gt record for one held-out paper."""
    rv = data.reviews().filter(pl.col("forum") == forum).sort("cdate").rows(named=True)
    cm = data.comments().filter((pl.col("forum") == forum) & (pl.col("role") == "Reviewer")).sort("cdate")
    items, reviewers = [], []
    for k, r in enumerate(rv):
        name = f"R{k + 1}"
        ws = extract(r)
        follow = "\n\n".join(c["text"] for c in cm.filter(pl.col("sig") == r["sig"]).iter_rows(named=True))
        status = resolve(ws, follow) if follow and ws else ["unknown"] * len(ws)
        for w, s in zip(ws, status):
            items.append(dict(text=w["text"], axis=w["axis"], reviewer=name, status=s))
        reviewers.append(dict(name=name, sig=r["sig"], scores={a: r[a] for a in AXES + ["overall"]},
                              followup=bool(follow)))
    return dict(forum=forum, reviewers=reviewers, items=items)


def dedup(items, thr):
    """greedy clustering, as in evaluation/extract_weaknesses.dedup.

    an item joins the most similar cluster seed if cosine >= thr and the
    cluster has no item of this reviewer yet; else it seeds a new cluster.
    """
    from mald import embed
    if not items:
        return []
    v = embed.vecs([it["text"] for it in items], "minilm")
    cl = []  # (seed index, member indices, reviewers)
    for i, it in enumerate(items):
        best, bs = None, thr
        for c in cl:
            sim = float(v[i] @ v[c[0]])
            if sim >= bs and it["reviewer"] not in c[2]:
                best, bs = c, sim
        if best is None:
            cl.append((i, [i], {it["reviewer"]}))
        else:
            best[1].append(i)
            best[2].add(it["reviewer"])
    return [c[1] for c in cl]


def clusters(g, thr=0.8):
    """concerns of a gt record with consensus, deficit and rebuttal status."""
    items = g["items"]
    sc = {r["name"]: r["scores"] for r in g["reviewers"]}
    out = []
    for grp in dedup(items, thr):
        its = [items[i] for i in grp]
        axes = [it["axis"] for it in its]
        axis = max(set(axes), key=axes.count)
        raisers = sorted({it["reviewer"] for it in its})
        defi = sum(1 - (sc[r][axis] if sc[r].get(axis) is not None else sc[r]["overall"]) for r in raisers)
        st = [it["status"] for it in its]
        out.append(dict(items=grp, texts=[it["text"] for it in its], axis=axis, reviewers=raisers,
                        consensus=len(raisers), deficit=defi,
                        resolved="resolved" in st, open="open" in st and "resolved" not in st))
    return out


def build(split, n="all", workers=4):
    """gt records of a split --> out/gt_<split>.jsonl (a rerun adds missing papers)."""
    from mald.paths import OUT
    out = OUT / f"gt_{split}.jsonl"
    from mald.run import papers
    from mald import jl
    done = {r["forum"] for r in jl.repair(out)} if os.path.exists(out) else set()
    todo = [x for x in papers(split, n) if x not in done]
    with ThreadPoolExecutor(workers) as ex, open(out, "a") as fh:
        for rec in ex.map(paper, todo):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()


if __name__ == "__main__":
    import sys
    build(*sys.argv[1:3])
