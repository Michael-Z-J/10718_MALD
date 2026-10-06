"""Download PDFs for a random sample of papers in data/papers.jsonl and extract their text.

Writes:
  data/sample_<year>_<n>.json   the sampled paper_ids (pass to match.py --paper-ids so every
                                system is scored on the same papers)
  data/pdfs/<paper_id>.pdf      raw PDFs (cached; re-runs skip papers already downloaded)
  data/fulltext/<paper_id>.txt  extracted text, pages separated by blank lines

Usage:
  pip install openreview-py pymupdf
  export OPENREVIEW_USERNAME=... OPENREVIEW_PASSWORD=...   # OpenReview blocks anonymous PDF requests
  python fetch_pdfs.py --year 2026 --n 50
"""

import argparse
import json
import os
import random
import re
import time
from datetime import datetime, timezone

from common import DATA_DIR, read_jsonl

PDF_DIR = os.path.join(DATA_DIR, "pdfs")
TEXT_DIR = os.path.join(DATA_DIR, "fulltext")
RESET_TIME = re.compile(r"'resetTime': '([^']+)'")


def sample_ids(year, n, seed):
    ids = sorted(p["paper_id"] for p in read_jsonl(os.path.join(DATA_DIR, "papers.jsonl")) if p["year"] == year)
    return sorted(random.Random(seed).sample(ids, min(n, len(ids))))


def client():
    import openreview
    if not os.environ.get("OPENREVIEW_USERNAME"):
        raise SystemExit("OpenReview blocks anonymous PDF requests with a challenge; "
                         "set OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD.")
    return openreview.api.OpenReviewClient(
        baseurl="https://api2.openreview.net",
        username=os.environ["OPENREVIEW_USERNAME"],
        password=os.environ.get("OPENREVIEW_PASSWORD"),
    )


def download(c, paper_id, retries=5):
    path = os.path.join(PDF_DIR, f"{paper_id}.pdf")
    if os.path.exists(path):
        return path
    for attempt in range(retries):
        try:
            data = c.get_pdf(paper_id)
            break
        except Exception as e:  # rate limits and transient errors: back off and retry
            reset = RESET_TIME.search(str(e))
            if reset:  # OpenReview allows ~26 PDFs per hour; wait out the window instead of failing
                wait = (datetime.fromisoformat(reset.group(1).replace("Z", "+00:00"))
                        - datetime.now(timezone.utc)).total_seconds() + 10
                print(f"  rate limited; waiting {wait / 60:.0f} min for the limit to reset")
                time.sleep(max(wait, 0))
                continue
            if attempt == retries - 1:
                print(f"  {paper_id}: giving up ({e})")
                return None
            time.sleep(2 ** attempt)
    if not data.startswith(b"%PDF"):  # e.g. a 404 error message instead of a file
        print(f"  {paper_id}: not a PDF ({data[:80]!r})")
        return None
    with open(path, "wb") as f:
        f.write(data)
    return path


def extract_text(pdf_path):
    import pymupdf
    with pymupdf.open(pdf_path) as doc:
        return "\n\n".join(page.get_text() for page in doc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ids = sample_ids(args.year, args.n, args.seed)
    sample_path = os.path.join(DATA_DIR, f"sample_{args.year}_{args.n}.json")
    with open(sample_path, "w") as f:
        json.dump(ids, f)
    print(f"Sampled {len(ids)} ICLR {args.year} papers -> {sample_path}")

    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(TEXT_DIR, exist_ok=True)
    c = client()
    failed = []
    for i, pid in enumerate(ids, 1):
        pdf = download(c, pid)
        if pdf is None:
            failed.append(pid)
            continue
        txt = os.path.join(TEXT_DIR, f"{pid}.txt")
        if not os.path.exists(txt):
            with open(txt, "w", encoding="utf-8") as f:
                f.write(extract_text(pdf))
        if i % 10 == 0:
            print(f"  {i}/{len(ids)}")
    print(f"Done: {len(ids) - len(failed)} texts in {TEXT_DIR}, {len(failed)} failed {failed[:10]}")


if __name__ == "__main__":
    main()
