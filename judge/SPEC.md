# Judge spec (identical for every judge)

You are judging outputs of {who} for an ICLR 2023 submission, against the paper itself and the real reviewers' points. Do not try to guess which system is which.

## Input
`<JUDGE_DIR>/in/<forum>.json` has:
- `paper_text`: path to the paper's first submitted version (plain text). Read all of it (in chunks if long) before judging.
- `real_items`: atomic weaknesses the real reviewers raised (`id`, `reviewer`, `axis`, `text`).
- `systems.<letter>.issues`: the system's ranked meta-review issues (`id` 0 = ranked first).
- `systems.<letter>.bots`: the system's simulated reviewers, each with a list of weaknesses.

## Labels
For every issue of every system:
- `valid`: "yes" if this paper really has this specific problem, judged from the paper text; "partial" if the problem is real but vague, generic, or partly inaccurate; "no" if it is not true of this paper, is boilerplate about some other paper, or misreads the paper.
- `importance`: 3 = could plausibly change a reviewer's accept/reject decision; 2 = a notable weakness; 1 = minor (use 1 when `valid` is "no").
- `doable`: true if the authors could address it within about four weeks before a deadline (writing, clarifications, modest extra experiments or analyses, added baselines that are available); false if it needs a new method, large-scale compute, new data collection, or a fundamental change of the contribution.
- `matches`: ids of `real_items` that state substantially the same specific concern (the same problem, not just the same topic). Empty list if none.

For every bot of every system:
- `covers`: ids of `real_items` that at least one of the bot's weaknesses states substantially the same concern as.

For every real item:
- `doable`: same definition as above.

## Output
Write exactly this JSON to `<JUDGE_DIR>/out/<forum>.json` (valid JSON, no comments). Include every issue id, every bot id and every real item id exactly once, in id order.
```
{"forum": "<forum>",
 "systems": {"A": {"issues": [{"id": 0, "valid": "yes", "importance": 3, "doable": true, "matches": [4, 17]}, ...],
                   "bots": [{"id": 0, "covers": [4, 9]}, ...]},
             "B": {...}, "C": {...}},
 "real_items": [{"id": 0, "doable": true}, ...]}
```
A system with no issues gets `"issues": []`. Be consistent and conservative: when unsure whether two points are the same concern, decide by whether fixing one would also fix the other.
