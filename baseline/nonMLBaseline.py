"""
Non-ML baseline: reviewers raise the same issues on similar papers.

For each test paper:
  1. Retrieve the k (default 10) most similar training papers by TF-IDF cosine similarity of
     title + abstract (--retrieval word2vec: averaged pretrained word2vec vectors instead).
  2. Rating = similarity-weighted mean of the neighbors' reviewer recommendations (1-10).
  3. Extract weakness sentences from the neighbors' reviews with rules (section headers
     like "Weaknesses:" / "Cons:" plus critical cue words), map each to a fixed weakness
     category via keyword lists, and score every category by the similarity-weighted
     fraction of neighbor reviewers who raised it.
  4. Severity (1-5) of each category = how harshly the neighbor reviewers who raised it scored
     their paper: round(8 - similarity-weighted mean recommendation), clipped to 1..5, so a mean
     recommendation of 3 or below is severity 5 and 7 or above is severity 1.
  5. By default (--rank_by severity) the categories the neighbors raised are ranked by this
     severity. Categories tied on severity at the 5th-place cutoff are recorded in
     "issue_ties", and evaluation/match.py scores the expected value over all equally likely
     ways to break the tie; "issues" shows one of them (the most frequent tied categories).
     --rank_by frequency ranks by the step-3 score instead.
  6. Each of the top 5 categories is output as a real reviewer sentence: among the neighbors'
     sentences in that category, the one most TF-IDF-similar to the test paper's title +
     abstract, preferring sentences with a critical cue word, skipping pasted BibTeX/LaTeX, and
     never reusing a sentence across categories (--issue_text template: the category's fixed
     suggested change instead).

Nothing is trained. Neighbors come from OpenReview ICLR years (--train_years, from
evaluation/fetch_reviews.py) or, by default, the raw allenai/PeerRead ICLR 2017 JSON, which is
fetched from GitHub on first run.

Writes one JSON line per paper ({"id", "title", "rating", "decision", "changes", "neighbors"}).
With --test_papers, the test set is instead OpenReview papers (evaluation/data/papers.jsonl), and
each line also has the fields evaluation/match.py reads:
{"paper_id", "issues": [5 changes], "severities", "issue_ties": {"fixed", "pool", "k"} (if tied)}.

Usage:
    pip install numpy scikit-learn            # + gensim for --retrieval word2vec
    python nonMLBaseline.py --test_papers ../evaluation/data/papers.jsonl --test_year 2026 --train_years 2024
    cd ../evaluation && python match.py outputs/nonml_iclr2026_severity_tfidf_quote_k10_train2024.jsonl \\
        --matcher embedding
"""

import argparse
import json
import os
import re
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(HERE, "data", "iclr_2017")
GITHUB_LIST_URL = "https://api.github.com/repos/allenai/PeerRead/contents/data/iclr_2017/{split}/reviews"
GITHUB_RAW_URL = "https://raw.githubusercontent.com/allenai/PeerRead/master/data/iclr_2017/{split}/reviews/{name}"
SPLITS = ("train", "dev", "test")


# ---------------------------------------------------------------- data loading

def _fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "nonMLBaseline"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def download_split(data_dir, split):
    out_dir = os.path.join(data_dir, split, "reviews")
    if os.path.isdir(out_dir) and any(f.endswith(".json") for f in os.listdir(out_dir)):
        return
    os.makedirs(out_dir, exist_ok=True)
    listing = json.loads(_fetch(GITHUB_LIST_URL.format(split=split)))
    names = [x["name"] for x in listing if x["name"].endswith(".json")]
    print(f"Downloading {len(names)} ICLR 2017 {split} review files -> {out_dir}")

    def save(name):
        data = _fetch(GITHUB_RAW_URL.format(split=split, name=name))
        with open(os.path.join(out_dir, name), "wb") as f:
            f.write(data)

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(save, names))


def _official_reviews(raw_reviews):
    """Keep reviewer reviews with a numeric recommendation; drop meta-reviews,
    PC decisions, public comments and the duplicated entries PeerRead contains."""
    seen, out = set(), []
    for r in raw_reviews:
        if str(r.get("IS_META_REVIEW")) == "True":
            continue
        try:
            rec = int(r.get("RECOMMENDATION"))
        except (TypeError, ValueError):
            continue
        comments = (r.get("comments") or "").strip()
        key = (r.get("OTHER_KEYS"), comments)
        if key in seen:
            continue
        seen.add(key)
        out.append({"recommendation": rec, "comments": comments})
    return out


def load_split(data_dir, split):
    download_split(data_dir, split)
    review_dir = os.path.join(data_dir, split, "reviews")
    papers = []
    for name in sorted(os.listdir(review_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(review_dir, name), encoding="utf-8", errors="replace") as f:
            d = json.load(f)
        reviews = _official_reviews(d.get("reviews", []))
        papers.append({
            "id": str(d.get("id", name[:-5])),
            "title": d.get("title", "") or "",
            "abstract": d.get("abstract", "") or "",
            "accepted": d.get("accepted"),
            "reviews": reviews,
            "rating": float(np.mean([r["recommendation"] for r in reviews])) if reviews else None,
        })
    return papers


def load_openreview_train(data_dir, years):
    """Neighbor pool from evaluation/fetch_reviews.py output. OpenReview reviews have a dedicated
    weaknesses field; it is prefixed with a "Weaknesses:" header so the rule-based extractor treats
    every sentence as a weakness, not only those with cue words."""
    with open(os.path.join(data_dir, "papers.jsonl"), encoding="utf-8") as f:
        papers = {r["paper_id"]: r for r in map(json.loads, f) if r["year"] in years}
    reviews = {}
    with open(os.path.join(data_dir, "reviews.jsonl"), encoding="utf-8") as f:
        for r in map(json.loads, f):
            if r["paper_id"] in papers and r["rating"] is not None and r["weaknesses"].strip():
                reviews.setdefault(r["paper_id"], []).append({
                    "recommendation": int(str(r["rating"]).split(":")[0]),  # 2024: "5: marginally below..."
                    "comments": "Weaknesses:\n" + r["weaknesses"]})
    missing = set(years) - {p["year"] for p in papers.values()}
    if missing:
        raise SystemExit(f"No ICLR {sorted(missing)} papers in {data_dir}: "
                         f"run evaluation/fetch_reviews.py --years <all years you need>")
    return [{"id": pid, "title": p["title"] or "", "abstract": p["abstract"] or "", "reviews": reviews[pid],
             "rating": float(np.mean([r["recommendation"] for r in reviews[pid]]))}
            for pid, p in papers.items() if pid in reviews]


def load_openreview_papers(path, year=None):
    """Test papers from evaluation/fetch_reviews.py output; reviews stay in evaluation/data."""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [{"id": r["paper_id"], "title": r["title"] or "", "abstract": r["abstract"] or ""}
            for r in rows if year is None or r["year"] == year]


# ---------------------------------------------------------------- embeddings

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have
having he her here hers herself him himself his how i if in into is it its itself just me more most
my myself no nor not now of off on once only or other our ours ourselves out over own same she should
so some such than that the their theirs them themselves then there these they this those through to
too under until up very was we were what when where which while who whom why will with would you your
yours yourself yourselves also however paper propose proposed show using use used via et al
""".split())

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]+")


def load_word2vec(name_or_path):
    from gensim.models import KeyedVectors
    if os.path.exists(name_or_path):
        if name_or_path.endswith(".kv"):
            return KeyedVectors.load(name_or_path)
        return KeyedVectors.load_word2vec_format(name_or_path, binary=name_or_path.endswith(".bin"))
    import gensim.downloader
    print(f"Loading word vectors '{name_or_path}' (downloaded and cached by gensim on first use)...")
    return gensim.downloader.load(name_or_path)


def embed(text, kv):
    vecs = []
    for tok in TOKEN_RE.findall(text):
        if tok.lower() in STOPWORDS:
            continue
        for cand in (tok, tok.lower()):
            if cand in kv.key_to_index:
                vecs.append(kv[cand])
                break
    if not vecs:
        return np.zeros(kv.vector_size, dtype=np.float32)
    v = np.mean(vecs, axis=0)
    return v / (np.linalg.norm(v) + 1e-12)


def paper_text(paper):
    return f"{paper['title']}. {paper['abstract']}"


# ---------------------------------------------------------------- weakness extraction

NEG_HEADER = re.compile(
    r"^\W*(weakness(es)?|cons|negatives?|concerns?|limitations?|criticisms?|drawbacks?|"
    r"(major|minor|other) (comments|issues|concerns|points|remarks|weaknesses))\b\s*:?",
    re.I)
POS_HEADER = re.compile(r"^\W*(strengths?|pros|positives?|summary|overall|contributions?)\b\s*:?", re.I)

WEAKNESS_CUES = re.compile(
    r"\b(lack\w*|missing|unclear|not (very |entirely |fully |really )?(clear|convinc\w*|obvious)|"
    r"unconvinc\w*|insufficient\w*|limited|weak\w*|concern\w*|issues?|problem\w*|confus\w*|"
    r"(hard|difficult) to (follow|read|understand)|should|"
    r"would (be|have been) (nice|good|better|helpful|interesting|useful)|could be (improved|better|clearer)|"
    r"i would (like|have liked) to see|not enough|fails? to|questionable|overclaim\w*|incremental|"
    r"unfortunately|however|no comparisons?|(does not|doesn't|did not|didn't) compare|not compared|"
    r"why not|drawbacks?|shortcomings?|downsides?|limitations?)\b",
    re.I)

# BibTeX / LaTeX that reviewers paste in (e.g. "abstract = {We revisit ...", "\citep{...}"):
# full of topic words, so it would win TF-IDF quote selection, but it isn't a weakness.
CITATION_JUNK = re.compile(r"^\s*\w+\s*=\s*\{|\\cite\w*\{|@\w+\{|\}\s*,\s*$")

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'\d])")

# (key, actionable change, keyword pattern). Order is the tie-break priority.
CATEGORIES = [
    ("baselines", "Compare against stronger, state-of-the-art baselines and prior methods.",
     r"baselines?|compar\w*|state[- ]of[- ]the[- ]art|sota|prior (work|methods?|approach\w*)|competing|outperform\w*"),
    ("experiments", "Strengthen the experiments: larger / real-world datasets, more tasks, more thorough evaluation.",
     r"experiment\w*|empirical\w*|evaluat\w*|datasets?|benchmarks?|toy|small[- ]scale|real[- ]world|mnist|cifar|imagenet"),
    ("clarity", "Improve clarity of writing, notation, and the explanation of the method.",
     r"clarity|unclear|not clear|(hard|difficult) to (follow|read|understand)|confus\w*|writing|written|presentation|"
     r"notation|typos?|explain\w*|explanation|descri\w*|readab\w*|organi[sz]\w*|figures?"),
    ("novelty", "Clarify and strengthen the novelty of the contribution relative to existing work.",
     r"novel\w*|original\w*|incremental|not new|already (been )?(proposed|known|done|explored)|contributions?|marginal"),
    ("soundness", "Justify the claims more rigorously (theory, assumptions, proofs, statistical support).",
     r"proofs?|prov(e|es|ed|en|ing)|theor\w*|claims?|justif\w*|assumptions?|sound\w*|correct\w*|rigou?r\w*|derivation|guarantees?|bounds?|"
     r"significance test|variance|error bars"),
    ("analysis", "Add ablations and analysis explaining why the method works (components, hyperparameters, sensitivity).",
     r"ablations?|analy[sz]\w*|sensitiv\w*|hyper-?parameters?|components?|insights?|intuitions?|understand\w*"),
    ("related_work", "Expand the related work discussion and cite missing relevant papers.",
     r"related work|cit(e|ed|ation\w*)|literature|references?|missing (work|papers)|previous work"),
    ("reproducibility", "Provide implementation details and code to make the results reproducible.",
     r"reproduc\w*|replicab\w*|code|implementation|details?|release\w*|training procedure|architecture details"),
    ("motivation", "Better motivate the problem and its practical significance / applicability.",
     r"motivat\w*|significan\w*|impact\w*|practical\w*|applicab\w*|useful\w*|relevan\w*|real applications?"),
    ("efficiency", "Discuss computational cost and scalability (runtime, memory, complexity).",
     r"computation\w*|cost\w*|run-?time|scalab\w*|scal(e|es|ing) (up|to|with|poorly|well)|efficien\w*|memory|speed|complexity|expensive|slow"),
]
CATEGORY_RES = [(key, action, re.compile(r"\b(" + pat + r")\b", re.I)) for key, action, pat in CATEGORIES]
ACTIONS = {key: action for key, action, _ in CATEGORIES}


def weakness_sentences(comments):
    """Rule-based extraction: everything under a weakness-style header, plus any
    sentence elsewhere that contains a critical cue word."""
    out, in_neg = [], False
    for line in comments.split("\n"):
        line = line.strip()
        if not line:
            continue
        m_neg, m_pos = NEG_HEADER.match(line), POS_HEADER.match(line)
        if m_neg:
            in_neg, line = True, line[m_neg.end():].strip()
        elif m_pos:
            in_neg, line = False, line[m_pos.end():].strip()
        for sent in SENT_SPLIT.split(line):
            sent = sent.strip(" -*\t")
            if not 4 <= len(sent.split()) <= 80:
                continue
            if in_neg or WEAKNESS_CUES.search(sent):
                out.append(sent)
    return out


def categorize(sentence):
    return [key for key, _, rx in CATEGORY_RES if rx.search(sentence)]


def review_weaknesses(review):
    """Map one review to {category: [weakness sentences]}."""
    by_cat = {}
    for sent in weakness_sentences(review["comments"]):
        for cat in categorize(sent):
            by_cat.setdefault(cat, []).append(sent)
    return by_cat


def category_prior(weaknesses):
    """Categories ordered by how many reviews raise them (ties keep the CATEGORIES order),
    given an iterable of review_weaknesses() dicts."""
    counts = Counter(c for w in weaknesses for c in w)
    return sorted(ACTIONS, key=lambda c: -counts[c])


# ---------------------------------------------------------------- the baseline

def rating_to_decision(score):
    s = round(score)
    if s <= 2:
        return "strong reject"
    if s <= 4:
        return "reject"
    if s == 5:
        return "borderline (weak reject)"
    if s == 6:
        return "borderline (weak accept)"
    if s <= 8:
        return "accept"
    return "strong accept"


def severity(mean_recommendation):
    """1 (minor) .. 5 (severe): recommendation <= 3 -> 5, 4 -> 4, 5 -> 3, 6 -> 2, >= 7 -> 1."""
    return int(np.clip(round(8 - mean_recommendation), 1, 5))


class NonMLReviewer:
    def __init__(self, train_papers, kv=None, k=10, rank_by="severity", retrieval="tfidf", issue_text="quote"):
        self.kv, self.k, self.rank_by, self.retrieval, self.issue_text = kv, k, rank_by, retrieval, issue_text
        self.train = [p for p in train_papers if p["reviews"]]
        # TF-IDF is fit on the training titles + abstracts; it also scores which reviewer sentence
        # to quote, so a quote sharing distinctive terms with the test paper is preferred.
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2, ngram_range=(1, 2))
        train_tfidf = self.tfidf.fit_transform([paper_text(p) for p in self.train])
        if retrieval == "tfidf":
            self.train_emb = train_tfidf
        else:
            self.train_emb = np.stack([embed(paper_text(p), kv) for p in self.train])
        for p in self.train:
            p["weaknesses"] = [review_weaknesses(r) for r in p["reviews"]]
        # Global category frequency over training reviews: only used to fill when
        # the neighbors raise fewer than 5 distinct categories.
        self.category_prior = category_prior(w for p in self.train for w in p["weaknesses"])
        # Mean recommendation of training reviews raising each category: the severity
        # fallback for categories no neighbor reviewer raised.
        recs = [(r["recommendation"], w) for p in self.train for r, w in zip(p["reviews"], p["weaknesses"])]
        overall = np.mean([rec for rec, _ in recs])
        self.category_rec = {c: np.mean([rec for rec, w in recs if c in w] or [overall]) for c in ACTIONS}

    def neighbors(self, text):
        if self.retrieval == "tfidf":
            sims = (self.train_emb @ self.tfidf.transform([text]).T).toarray().ravel()
        else:
            sims = self.train_emb @ embed(text, self.kv)
        idx = np.argsort(-sims)[: self.k]
        return [(self.train[i], float(sims[i])) for i in idx]

    def pick_quotes(self, ranked, candidates, text):
        """For each category in rank order, the neighbor reviewer sentence most TF-IDF-similar to
        the test paper, never reusing a sentence already chosen for a higher-ranked category
        (a sentence can fall under several categories). Ties go to the closest neighbor's sentence."""
        q = self.tfidf.transform([text])
        chosen, used = {}, set()
        for cat in ranked:
            cands = [(sent, pid) for sent, pid in candidates.get(cat, [])
                     if sent not in used and not CITATION_JUNK.search(sent)]
            # Prefer sentences that actually criticize something; topical but neutral sentences
            # ("The experiments showed ...") would otherwise win on TF-IDF overlap.
            cands = [c for c in cands if WEAKNESS_CUES.search(c[0])] or cands
            if not cands:
                continue
            sims = (self.tfidf.transform([sent for sent, _ in cands]) @ q.T).toarray().ravel()
            best = cands[int(np.argmax(sims))]  # argmax keeps the first, i.e. closest neighbor, on ties
            chosen[cat] = best
            used.add(best[0])
        return chosen

    def review(self, title, abstract, n_changes=5):
        text = f"{title}. {abstract}"
        nbrs = self.neighbors(text)
        weights = np.array([max(s, 0.0) for _, s in nbrs])
        if weights.sum() <= 0:
            weights = np.ones(len(nbrs))
        weights = weights / weights.sum()

        rating = float(sum(w * p["rating"] for w, (p, _) in zip(weights, nbrs)))

        # Score = similarity-weighted fraction of each neighbor's reviewers raising the category;
        # the same weights average those reviewers' recommendations for the severity.
        scores, rec_sum, candidates = Counter(), Counter(), {}
        for w, (p, _) in zip(weights, nbrs):
            n_rev = len(p["weaknesses"])
            for review, rev in zip(p["reviews"], p["weaknesses"]):
                for cat, sents in rev.items():
                    scores[cat] += w / n_rev
                    rec_sum[cat] += w / n_rev * review["recommendation"]
                    # nbrs is sorted by similarity, so candidates run from the closest neighbor
                    candidates.setdefault(cat, []).extend((sent, p["id"]) for sent in sents)
        mean_rec = {c: rec_sum[c] / scores[c] if scores[c] > 0 else self.category_rec[c] for c in ACTIONS}

        order = {c: i for i, c in enumerate(self.category_prior)}
        by_frequency = lambda c: (-scores[c], order[c])  # noqa: E731
        ties = None
        if self.rank_by == "severity":
            sev = {c: severity(mean_rec[c]) for c in scores}
            ranked = sorted(scores, key=lambda c: (-sev[c], by_frequency(c)))
            if len(ranked) > n_changes:
                cutoff = sev[ranked[n_changes - 1]]
                fixed = [c for c in ranked if sev[c] > cutoff]
                pool = [c for c in ranked if sev[c] == cutoff]
                if len(fixed) + len(pool) > n_changes:
                    ties = {"fixed": fixed, "pool": pool, "k": n_changes - len(fixed)}
        else:
            ranked = sorted(scores, key=by_frequency)
        ranked += [c for c in self.category_prior if c not in ranked]

        # Every category that could enter the top 5 (including the whole tied pool) needs its text.
        quotes = self.pick_quotes(ranked, candidates, text) if self.issue_text == "quote" else {}
        issue = {c: quotes[c][0] if c in quotes else ACTIONS[c] for c in ranked}
        if ties:
            ties = {"fixed": [issue[c] for c in ties["fixed"]], "pool": [issue[c] for c in ties["pool"]],
                    "k": ties["k"]}
        changes = []
        for cat in ranked[:n_changes]:
            changes.append({"category": cat, "change": issue[cat], "template": ACTIONS[cat],
                            "score": round(scores[cat], 4), "severity": severity(mean_rec[cat]),
                            "mean_neighbor_rec": round(mean_rec[cat], 2),
                            "source_paper_id": quotes[cat][1] if cat in quotes else None})

        return {
            "issue_ties": ties,
            "rating": round(rating, 2),
            "decision": rating_to_decision(rating),
            "changes": changes,
            "neighbors": [{"id": p["id"], "title": p["title"], "similarity": round(s, 4),
                           "mean_rating": round(p["rating"], 2)} for p, s in nbrs],
        }


# ---------------------------------------------------------------- output

def print_review(r):
    print("=" * 100)
    print(f"[{r['id']}] {r['title']}")
    print(f"  predicted rating: {r['rating']:.2f} ({r['decision']})")
    print("  nearest training papers:")
    for n in r["neighbors"]:
        print(f"    sim={n['similarity']:.3f}  rating={n['mean_rating']:.2f}  {n['title'][:80]}")
    print("  5 suggested changes:")
    for i, c in enumerate(r["changes"], 1):
        src = f" (reviewer of {c['source_paper_id']})" if c["source_paper_id"] else ""
        print(f"    {i}. [{c['category']}, score={c['score']:.3f}, severity={c['severity']}]{src} {c['change'][:200]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--k", type=int, default=10, help="number of nearest training papers")
    ap.add_argument("--retrieval", default="tfidf", choices=["tfidf", "word2vec"],
                    help="how neighbors are found from title + abstract")
    ap.add_argument("--issue_text", default="quote", choices=["quote", "template"],
                    help="each issue is the most relevant neighbor reviewer sentence in its category, "
                         "or the category's fixed suggested change")
    ap.add_argument("--rank_by", default="severity", choices=["severity", "frequency"],
                    help="order the 5 changes by severity (ties at the cutoff scored in expectation), "
                         "or by how often neighbors raised them")
    ap.add_argument("--w2v", default="word2vec-google-news-300",
                    help="gensim-downloader model name, or path to a .bin/.txt/.kv word2vec file")
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                    help="dir with {train,dev,test}/reviews/*.json (e.g. PeerRead/data/iclr_2017)")
    ap.add_argument("--show", type=int, default=3, help="number of example reviews to print")
    ap.add_argument("--test_papers", default=None,
                    help="OpenReview papers.jsonl to review instead of the PeerRead split")
    ap.add_argument("--test_year", type=int, default=None, help="only review --test_papers from this year")
    ap.add_argument("--train_years", type=int, nargs="+", default=None,
                    help="retrieve neighbors from these ICLR years in evaluation/data (e.g. 2024 2025) "
                         "instead of the PeerRead ICLR 2017 train split")
    ap.add_argument("--out", default=None, help="output JSONL path (default: baseline/nonml_iclr2017_<split>.jsonl, or "
                                                "evaluation/outputs/nonml_iclr<year>_<rank_by>.jsonl with --test_papers)")
    args = ap.parse_args()

    if args.train_years:
        train = load_openreview_train(os.path.join(HERE, "..", "evaluation", "data"), args.train_years)
        if args.test_year in args.train_years:
            raise SystemExit("--test_year must not be one of --train_years")
    else:
        train = load_split(args.data_dir, "train")
    if args.test_papers:
        test = load_openreview_papers(args.test_papers, args.test_year)
        test_name = f"OpenReview {args.test_year or 'all years'}"
        default_out = os.path.join(HERE, "..", "evaluation", "outputs",
                                   f"nonml_iclr{args.test_year or 'all'}_{args.rank_by}_{args.retrieval}"
                                   f"_{args.issue_text}_k{args.k}"
                                   + (f"_train{'-'.join(map(str, args.train_years))}" if args.train_years else "")
                                   + ".jsonl")
    else:
        test = load_split(args.data_dir, args.split)
        test_name = args.split
        default_out = os.path.join(HERE, f"nonml_iclr2017_{args.split}.jsonl")
    print(f"Loaded {len(train)} train / {len(test)} {test_name} papers "
          f"({sum(bool(p['reviews']) for p in train)} train papers with rated reviews)")

    kv = load_word2vec(args.w2v) if args.retrieval == "word2vec" else None
    reviewer = NonMLReviewer(train, kv, k=args.k, rank_by=args.rank_by, retrieval=args.retrieval,
                             issue_text=args.issue_text)

    out_path = args.out or default_out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, p in enumerate(test):
            r = {"id": p["id"], "title": p["title"], **reviewer.review(p["title"], p["abstract"])}
            if args.test_papers:
                r.update(paper_id=p["id"], issues=[c["change"] for c in r["changes"]],
                         severities=[c["severity"] for c in r["changes"]])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i < args.show:
                print_review(r)
    print("=" * 100)
    print(f"Wrote {len(test)} reviews to {out_path}")
    print(f"Evaluate with: cd evaluation && python match.py {os.path.relpath(out_path, os.path.join(HERE, '..', 'evaluation'))} --matcher embedding")


if __name__ == "__main__":
    main()
