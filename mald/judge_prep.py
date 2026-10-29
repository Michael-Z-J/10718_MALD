"""build blinded input bundles for the llm-judge subagents (secondary metrics).

for each of the first n test papers (seeded order of mald.run.papers):
  paper text path, real reviewer items (id, reviewer, axis, text),
  and per system: ranked meta issues (id, text) and panel bots (id, weakness texts)
--> judge/in/<forum>.json, with system names replaced by letters in a per-paper
    seeded order; the key to unblind is written to judge/key.json only.
the bundle carries no system flags or scores: the judge labels validity,
importance and doability blind, and the aggregation compares them to the flags.

the judge spec (judge/SPEC.md) is written to the judge dir with the dir path
and the number of systems filled in.

usage: MALD_JUDGE=<judge dir> python -m mald.judge_prep <n> <system> [<system> ...]
the letter shuffle depends on the system list: use one judge dir per list.
a rerun with a larger n and the same systems keeps the letters of earlier papers.
"""
import json, random, sys

from mald import jl, run
from mald.paths import BUILD, JUDGE as J, OUT, REPO

TEXT = BUILD / "text"
MAX_BOT = 8
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}


def spec(k):
    """judge/SPEC.md for k systems, with the judge dir filled in."""
    who = ("one anonymous automated reviewing system (letter A)" if k == 1 else
           f"{WORDS[k]} anonymous automated reviewing systems (letters {', '.join('ABCDEFG'[:k])})")
    s = (REPO / "judge" / "SPEC.md").read_text().replace("{who}", who).replace("<JUDGE_DIR>", str(J))
    (J / "SPEC.md").write_text(s)


def main(n, *systems):
    (J / "in").mkdir(parents=True, exist_ok=True)
    (J / "out").mkdir(parents=True, exist_ok=True)
    gts = {r["forum"]: r for r in jl.rows(OUT / "gt_test.jsonl")}
    recs = {s: {r["forum"]: r for r in jl.rows(OUT / "test" / f"{s}.jsonl")} for s in systems}
    key = {}
    for fid in run.papers("test", n):
        g = gts[fid]
        order = list(systems)
        random.Random(f"judge-{fid}").shuffle(order)
        bundle = dict(forum=fid, paper_text=str(TEXT / f"{fid}.txt"),
                      real_items=[dict(id=i, reviewer=it["reviewer"], axis=it["axis"], text=it["text"])
                                  for i, it in enumerate(g["items"])],
                      systems={})
        key[fid] = {}
        for letter, s in zip("ABCDEFG", order):
            r = recs[s][fid]
            bundle["systems"][letter] = dict(
                issues=[dict(id=i, text=x["issue"]) for i, x in enumerate(r["meta"]["issues"])],
                bots=[dict(id=b, weaknesses=[w["issue"] for w in bot["weaknesses"][:MAX_BOT]])
                      for b, bot in enumerate(r["panel"])])
            key[fid][letter] = s
        (J / "in" / f"{fid}.json").write_text(json.dumps(bundle, indent=1))
    (J / "key.json").write_text(json.dumps(key, indent=1))
    spec(len(systems))
    print(len(key), "bundles;", {f: len(json.loads((J / "in" / f"{f}.json").read_text())["real_items"]) for f in key})


if __name__ == "__main__":
    main(int(sys.argv[1]), *sys.argv[2:])
