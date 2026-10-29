"""model selection on val by ndcg_cons (tie-break recall_w).
both are embedding-matched values pooled over gt grouping 0.30-0.80 and
tau 0.40-0.80. the llm judge is not used for selection.

out/val/summary.json --> argmax over every method variant run on val:
  m_full / m_unc / m_noret / m_full_nemo      (panel: generator, retrieval, modes; llm meta, theta 0.2)
  det_t*_w*                                   (gemma panel, deterministic meta: theta x w grid)
  det_nemo                                    (nemotron panel, deterministic meta at the best det theta, w)
  m_flags / m_nomodes / m_norubric / m_noflags (gemma panel, llm meta at the best det theta, w)
and b1 vs b1_nemo for the generator of ours (b1)
--> report/selected.json (method kwargs) + report/selected_b1.json + report/selection.json
"""
import json, re

from mald.paths import OUT, REPORT as REP
FIXED = {"m_full": {"theta": 0.2}, "m_unc": {"n_reb": 0, "theta": 0.2}, "m_noret": {"retrieval": False, "theta": 0.2},
         "m_full_nemo": {"model": "nemotron", "theta": 0.2}}
FLAGS = {"m_flags": {}, "m_nomodes": {"modes": False}, "m_norubric": {"rubric": False},
         "m_noflags": {"modes": False, "rubric": False}}


def key(s, n):
    return (s[n].get("ndcg_cons", -1), s[n].get("recall_w", -1))


def candidates(s):
    """name --> method kwargs for every method variant present in the summary."""
    dets = [n for n in s if re.match(r"det_t[\d.]+_w[\d.]+$", n)]
    best = max(dets, key=lambda n: key(s, n))
    t, w = map(float, re.match(r"det_t([\d.]+)_w([\d.]+)", best).groups())
    out = {n: kw for n, kw in FIXED.items() if n in s}
    out |= {n: dict(theta=float(a), w=float(b), llm_meta=False)
            for n in dets for a, b in [re.match(r"det_t([\d.]+)_w([\d.]+)", n).groups()]}
    if "det_nemo" in s:
        out["det_nemo"] = dict(model="nemotron", theta=t, w=w, llm_meta=False)
    out |= {n: dict(theta=t, w=w, **kw) for n, kw in FLAGS.items() if n in s}
    return out


def main():
    s = json.loads((OUT / "val" / "summary.json").read_text())["systems"]
    cand = candidates(s)
    pick = max(cand, key=lambda n: key(s, n))
    b1 = max([n for n in ("b1", "b1_nemo") if n in s], key=lambda n: key(s, n))
    (REP / "selected.json").write_text(json.dumps(cand[pick]))
    (REP / "selected_b1.json").write_text(json.dumps({"model": "nemotron"} if b1 == "b1_nemo" else {}))
    sel = dict(pick=pick, kwargs=cand[pick], b1=b1,
               ranking=sorted(([n, s[n]["ndcg_cons"], s[n]["recall_w"]] for n in cand), key=lambda r: -r[1]),
               table={n: {k: s[n].get(k) for k in ("ndcg_cons", "recall_w", "recall_multi", "precision", "n_papers")} for n in s})
    (REP / "selection.json").write_text(json.dumps(sel, indent=1))
    print(pick, json.dumps(cand[pick]), "b1:", b1)


if __name__ == "__main__":
    main()
