"""build the corpus from the casimir release.

casimir files (download if missing) --> forums / reviews / comments tables
  --> rubric scaling --> dedup --> split (test / val / pool / concurrent / drop)
  --> first-version text of test and val papers --> parquet + txt + report json.

split rule (set before any model run):
  test = iclr 2023 forums in casimir mapping_small_test
  val  = iclr 2022 forums in casimir mapping_small_test
  pool = all other forums of a main venue before iclr 2023, minus resubmission
         twins of test and val. mald.data sets which pool years a target can use.
  concurrent = other iclr 2023 forums. their reviews were written in the same
         window as the test reviews, so they are never used.
  drop = twins, forums without reviews, other venues

an optional second corpus (json list of {forum_id, title}) is checked for
overlap with the splits; the counts go into the report.

usage: python -m mald.build [other.json]
"""
import json, re, sys, hashlib, collections as C
import urllib.request
from pathlib import Path

import polars as pl

from mald import paths, rubric

HF = "https://huggingface.co/datasets/taln-ls2n/CASIMIR/resolve/main/"
FILES = ("README.md", "reviews.jsonl", "metadata.jsonl", "mapping_full_corpus.jsonl",
         "mapping_small_test.jsonl", "article_pairs_small_test.jsonl")

MAIN = {"ICLR.cc/2020/Conference", "ICLR.cc/2021/Conference", "ICLR.cc/2022/Conference",
        "ICLR.cc/2023/Conference", "NeurIPS.cc/2022/Conference"}
TEXT_KEYS = ("summary_of_the_paper", "summary", "main_review", "review", "strength_and_weaknesses",
             "strengths_and_weaknesses", "strengths", "weaknesses", "questions", "limitations",
             "clarity,_quality,_novelty_and_reproducibility", "summary_of_the_review")


def fetch(root):
    """download the casimir files that are not in root yet (official hf release)."""
    root.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if not (root / f).exists():
            print("fetch", f, flush=True)
            urllib.request.urlretrieve(HF + f, root / f)


def venue(sig):
    return re.split(r"/(Paper|Submission)\d+|/Program_Chairs|/Area_Chair", sig)[0]


def role(sig):
    tail = sig.rsplit("/", 1)[-1]
    for k in ("Reviewer", "Authors", "Area_Chair", "Program_Chairs"):
        if tail.startswith(k) or tail.startswith("Anon" + k):
            return k
    return "other"


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def jread(p):
    with open(p) as f:
        for line in f:
            yield json.loads(line)


def notes(root):
    """casimir notes --> (reviews, comments, decisions) row lists."""
    revs, coms, decs = [], [], {}
    for r in jread(root / "reviews.jsonl"):
        sig, c = r["signatures"][0], r["content"]
        v, ro = venue(sig), role(sig)
        if "decision" in c:
            decs[r["forum"]] = str(c["decision"])
            continue
        raw = rubric.raw(c)
        text = "\n\n".join(f"{k}: {c[k]}" for k in TEXT_KEYS if isinstance(c.get(k), str) and c[k].strip())
        if ro == "Reviewer" and raw["overall"] is not None and text:
            revs.append(dict(forum=r["forum"], id=r["id"], venue=v, sig=sig, cdate=r["cdate"],
                             text=text, conf=rubric.num(c.get("confidence")), **{f"raw_{k}": x for k, x in raw.items()}))
        elif ro in ("Reviewer", "Authors", "Area_Chair"):
            body = c.get("comment") or c.get("metareview") or c.get("response") or ""
            if isinstance(body, str) and body.strip():
                coms.append(dict(forum=r["forum"], id=r["id"], replyto=r["replyto"], venue=v,
                                 sig=sig, role=ro, cdate=r["cdate"], text=body))
    return revs, coms, decs


def meta(root):
    """casimir metadata --> forum info + version dates."""
    info, vdate = {}, {}
    for r in jread(root / "metadata.jsonl"):
        c = r["content"]
        vdate[r["id"]] = r.get("tcdate") or r.get("cdate")
        if r["id"] == r["forum"] and c.get("title"):
            info[r["forum"]] = dict(title=c["title"], abstract=c.get("abstract", ""),
                                    authors=c.get("authorids") or c.get("authors") or [])
    return info, vdate


def v1_text(pairs, v1):
    """text of the first version: sentence-1 side of its pairs, in order.

    the first version is the text that the reviewers read.
    """
    rows = sorted((p for p in pairs if p["id_version_1"] == v1), key=lambda p: p["sentence-pair-index"])
    out, seen, sec = [], set(), None
    for p in rows:
        sid, t, ty = p.get("id-sentence-1"), p.get("text-sentence-1"), p.get("type-sentence-1")
        if not sid or not t or sid in seen:
            continue
        seen.add(sid)
        if ty in ("Title", "Section"):
            out.append(f"\n## {t}\n")
        elif ty == "Abstract":
            out.append(f"\n## Abstract\n{t}")
        else:
            out.append(t)
    return re.sub(r"\n{3,}", "\n\n", " ".join(out)).strip()


def overlap(other, fdf):
    """forum id and title overlap of a second corpus with each split."""
    if not other:
        return {}
    o = json.load(open(other))
    ids, titles = {x["forum_id"] for x in o}, {norm_title(x["title"]) for x in o}
    out = dict(other_id_overlap=len(ids & set(fdf["forum"])))
    for s in ("test", "val", "pool"):
        out[f"other_title_overlap_{s}"] = int(sum(norm_title(t) in titles for t in fdf.filter(pl.col("split") == s)["title"]))
    return out


def main(other=None):
    root, out = paths.CASIMIR, paths.BUILD
    fetch(root)
    out.mkdir(parents=True, exist_ok=True)
    revs, coms, decs = notes(root)
    info, vdate = meta(root)
    rng = rubric.ranges((r["venue"], {a: r[f"raw_{a}"] for a in rubric.AXES}) for r in revs)
    for r in revs:
        r.update(rubric.scale(r["venue"], {a: r[f"raw_{a}"] for a in rubric.AXES}, rng))

    # review dedup: (forum, signature) keeps latest; identical text dropped
    by, hashes, ndup = {}, set(), C.Counter()
    for r in sorted(revs, key=lambda r: r["cdate"]):
        h = hashlib.sha1(r["text"].encode()).hexdigest()
        if h in hashes:
            ndup["text"] += 1
            continue
        if (r["forum"], r["sig"]) in by:
            ndup["sig"] += 1
        hashes.add(h)
        by[(r["forum"], r["sig"])] = r
    revs = list(by.values())

    fven = {}
    for r in revs:
        fven.setdefault(r["forum"], r["venue"])
    small = {m["id_forum"]: m["references"] for m in jread(root / "mapping_small_test.jsonl")}
    full = {m["id_forum"]: m["references"] for m in jread(root / "mapping_full_corpus.jsonl")}

    forums = []
    for f, v in fven.items():
        d = decs.get(f, "")
        forums.append(dict(forum=f, venue=v, year=int(m.group()) if (m := re.search(r"\d{4}", v)) else 0,
                           title=info.get(f, {}).get("title", ""), abstract=info.get(f, {}).get("abstract", ""),
                           authors=list(info.get(f, {}).get("authors", [])), decision=d,
                           accepted=d.lower().startswith("accept"), n_versions=len(full.get(f, [])),
                           in_small=f in small))
    fdf = pl.DataFrame(forums, infer_schema_length=None)

    def pick(r):
        if r["venue"] not in MAIN:
            return "drop"
        if r["in_small"] and r["venue"] == "ICLR.cc/2023/Conference":
            return "test"
        if r["in_small"] and r["venue"] == "ICLR.cc/2022/Conference":
            return "val"
        if r["venue"] == "ICLR.cc/2023/Conference":
            return "concurrent"
        return "pool"

    fdf = fdf.with_columns(pl.Series("split", [pick(r) for r in fdf.iter_rows(named=True)]))

    # resubmission twins: same normalized title as a test/val paper --> drop from pool
    held = fdf.filter(pl.col("split").is_in(["test", "val"]))
    held_t = set(norm_title(t) for t in held["title"])
    twin = [r["split"] == "pool" and norm_title(r["title"]) in held_t for r in fdf.iter_rows(named=True)]
    fdf = fdf.with_columns(pl.when(pl.Series(twin)).then(pl.lit("drop")).otherwise(pl.col("split")).alias("split"))

    # tf-idf near-duplicate abstracts (resubmissions with new titles)
    from sklearn.feature_extraction.text import TfidfVectorizer
    pool = fdf.filter(pl.col("split") == "pool")
    held = fdf.filter(pl.col("split").is_in(["test", "val"]))
    vec = TfidfVectorizer(stop_words="english", min_df=2).fit(fdf["abstract"].to_list())
    sim = (vec.transform(held["abstract"].to_list()) @ vec.transform(pool["abstract"].to_list()).T).tocsr()
    near = set()
    for i in range(sim.shape[0]):
        row = sim.getrow(i)
        near.update(pool["forum"][int(j)] for j, s in zip(row.indices, row.data) if s >= 0.8)
    fdf = fdf.with_columns(pl.when(pl.col("forum").is_in(list(near))).then(pl.lit("drop"))
                           .otherwise(pl.col("split")).alias("split"))

    split = dict(zip(fdf["forum"], fdf["split"]))
    rdf = pl.DataFrame(revs, infer_schema_length=None).with_columns(pl.col("forum").replace_strict(split, default="drop").alias("split"))
    cdf = pl.DataFrame(coms, infer_schema_length=None).with_columns(pl.col("forum").replace_strict(split, default="drop").alias("split"))

    # first-version text of test and val papers
    tdir = out / "text"
    tdir.mkdir(exist_ok=True)
    want = {f: small[f] for f, s in split.items() if s in ("test", "val")}
    v1 = {f: min(refs, key=lambda x: int(vdate.get(x) or 9e15)) for f, refs in want.items()}
    by_v1 = C.defaultdict(list)
    inv = {v: f for f, v in v1.items()}
    for p in jread(root / "article_pairs_small_test.jsonl"):
        if p["id_version_1"] in inv:
            by_v1[p["id_version_1"]].append(p)
    ntext = 0
    for f, v in v1.items():
        t = v1_text(by_v1.get(v, []), v)
        if len(t) > 2000:
            (tdir / f"{f}.txt").write_text(t)
            ntext += 1
    has = {p.stem for p in tdir.glob("*.txt")}
    fdf = fdf.with_columns(pl.col("forum").is_in(list(has)).alias("has_text"),
                           pl.col("forum").replace_strict(v1, default=None).alias("v1"))

    fdf.write_parquet(out / "forums.parquet")
    rdf.write_parquet(out / "reviews.parquet")
    cdf.write_parquet(out / "comments.parquet")
    nrev = rdf.group_by("forum").len()
    rep = dict(
        n_reviews=len(rdf), review_dups=dict(ndup), splits=dict(C.Counter(split.values())),
        twins_title=int(sum(twin)), twins_abstract=len(near), texts=ntext,
        test_accept_rate=float(fdf.filter(pl.col("split") == "test")["accepted"].mean()),
        ranges={f"{v}|{a}": r for (v, a), r in rng.items() if v in MAIN},
        reviews_per_forum=nrev["len"].describe().to_dicts(), **overlap(other, fdf))
    (out / "build_report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main(*sys.argv[1:2])
