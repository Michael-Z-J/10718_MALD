# Evaluation: matching system issues to reviewer weaknesses (ICLR 2024+)

Every system (non-ML baseline, Qwen baseline, our model) writes one JSONL file:
`{"paper_id": "<OpenReview forum id>", "issues": ["...", ...]}` and goes through the same pipeline.

1. Fill in `call_llm()` in `common.py` (use a different model family from the LLM baseline).
2. `python fetch_reviews.py --years 2024 2025` → `data/papers.jsonl`, `data/reviews.jsonl`
3. `python extract_weaknesses.py` → `data/weaknesses.jsonl` (split + dedup across reviewers, `n_reviewers` weight, substantive/minor tag).
   Spot-check ~50 reviews by hand and report the error rate.
4. `python validate.py sample outputs/*.jsonl --n 200`; each annotator labels a copy of `data/labels/template.csv`,
   then `python validate.py score data/labels/labels_*.csv` (kappa between annotators, judge/embedding agreement on held-out half).
5. `python match.py outputs/*.jsonl --substantive-only` → recall@5 (headline), weighted recall@5, precision@5.


ICLR 2024: kept 5690 papers, dropped {'Withdrawn_Submission': 1661, 'Desk_Rejected_Submission': 53}
5690 papers, 22011 reviews