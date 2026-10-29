"""run one system config on a held-out split.

usage: python -m mald.run <split> <n|all> <name> <system> '<json kwargs>'
  --> out/<split>/<name>.jsonl, one record per paper. a rerun skips the
  papers that are already in the file.
papers: the first n of a seeded shuffle of the split, so all configs of a
split get the same papers.
"""
import json, random, sys, time, traceback

from mald import data, jl, systems
from mald.paths import OUT
YEAR = {"test": 2023, "val": 2022}


def papers(split, n):
    f, _ = data.held(split)
    ids = sorted(f["forum"].to_list())
    random.Random(718).shuffle(ids)
    return ids if n == "all" else ids[: int(n)]


def main(split, n, name, system, kw="{}"):
    kw = json.loads(kw)
    path = OUT / split / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["forum"] for r in jl.repair(path)} if path.exists() else set()
    fn = getattr(systems, system)
    with open(path, "a") as fh:
        for fid in papers(split, n):
            if fid in done:
                continue
            t = time.time()
            try:
                rec = fn(fid, YEAR[split], **kw)
            except Exception:
                traceback.print_exc()
                continue
            rec.update(forum=fid, system=name, kw=kw, secs=time.time() - t)
            fh.write(json.dumps(rec, default=float) + "\n")
            fh.flush()
            print(name, fid, f"{rec['secs']:.0f}s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
