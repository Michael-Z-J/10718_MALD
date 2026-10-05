"""Stage 0: download ICLR 2024+ submissions and official reviews from OpenReview (API v2).

Raw notes are cached once per venue in data/raw/; everything downstream works offline.
Writes:
  data/papers.jsonl   {paper_id, year, title, abstract}
  data/reviews.jsonl  {paper_id, reviewer, rating, weaknesses}

Usage:
  pip install openreview-py
  export OPENREVIEW_USERNAME=... OPENREVIEW_PASSWORD=...   # if anonymous access fails
  python fetch_reviews.py --years 2024 2025
"""

import argparse
import json
import os
from collections import Counter

from common import DATA_DIR, write_jsonl

DROP_VENUEIDS = ("Withdrawn_Submission", "Desk_Rejected_Submission")


def fetch_raw(year):
    path = os.path.join(DATA_DIR, "raw", f"iclr{year}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    import openreview
    if not os.environ.get("OPENREVIEW_USERNAME"):
        raise SystemExit("OpenReview blocks anonymous API requests with a challenge; "
                         "set OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD.")
    client = openreview.api.OpenReviewClient(
        baseurl="https://api2.openreview.net",
        username=os.environ["OPENREVIEW_USERNAME"],
        password=os.environ.get("OPENREVIEW_PASSWORD"),
    )
    venue = f"ICLR.cc/{year}/Conference"
    subs = client.get_all_notes(invitation=f"{venue}/-/Submission", details="replies")
    raw = [{"id": s.id, "content": s.content, "replies": s.details["replies"]} for s in subs]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(raw, f)
    return raw


def val(content, key):
    return (content.get(key) or {}).get("value")  # v2 wraps every field in {"value": ...}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2024])
    args = ap.parse_args()

    papers, reviews = [], []
    for year in args.years:
        dropped = Counter()
        for sub in fetch_raw(year):
            venueid = val(sub["content"], "venueid") or ""
            if venueid.endswith(DROP_VENUEIDS):
                dropped[venueid.rsplit("/", 1)[-1]] += 1
                continue
            revs = [r for r in sub["replies"]
                    if any(i.endswith("Official_Review") for i in r["invitations"])]
            if not revs:
                dropped["no_reviews"] += 1
                continue
            papers.append({"paper_id": sub["id"], "year": year,
                           "title": val(sub["content"], "title"),
                           "abstract": val(sub["content"], "abstract")})
            for r in revs:
                reviews.append({"paper_id": sub["id"], "reviewer": r["signatures"][0],
                                "rating": val(r["content"], "rating"),
                                "weaknesses": val(r["content"], "weaknesses") or ""})
        print(f"ICLR {year}: kept {sum(p['year'] == year for p in papers)} papers, dropped {dict(dropped)}")

    write_jsonl(os.path.join(DATA_DIR, "papers.jsonl"), papers)
    write_jsonl(os.path.join(DATA_DIR, "reviews.jsonl"), reviews)
    print(f"{len(papers)} papers, {len(reviews)} reviews -> {DATA_DIR}")


if __name__ == "__main__":
    main()
