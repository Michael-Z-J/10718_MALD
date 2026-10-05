"""
Non-ML baseline for PeerRead (ICLR 2017).

For each test paper:
  1. Embed title + abstract as the average of pretrained (frozen) word2vec vectors.
  2. Retrieve the k most similar training papers by cosine similarity.
  3. Rating = similarity-weighted mean of the neighbors' reviewer recommendations (1-10).
  4. Extract weakness sentences from the neighbors' reviews with rules (section headers
     like "Weaknesses:" / "Cons:" plus critical cue words), map each to a fixed weakness
     category via keyword lists, and score every category by the similarity-weighted
     fraction of neighbor reviewers who raised it. The top 5 categories are the 5 changes,
     each with a supporting quote from the most similar neighbor that raised it.

Nothing is trained on PeerRead. Data is the raw allenai/PeerRead JSON (the same files the
HF `allenai/peer_read` loader downloads); it is fetched from GitHub on first run.

Writes one JSON line per paper ({"id", "title", "rating", "decision", "changes", "neighbors"});
scoring against the real reviews is done by evaluation/eval.py.

Usage:
    pip install gensim numpy
    python nonMLBaseline.py                      # review the ICLR 2017 test split
    python nonMLBaseline.py --split dev --k 5 --show 3
    python evaluation/eval.py baseline/nonml_iclr2017_test.jsonl
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


class NonMLReviewer:
    def __init__(self, train_papers, kv, k=5):
        self.kv, self.k = kv, k
        self.train = [p for p in train_papers if p["reviews"]]
        self.train_emb = np.stack([embed(paper_text(p), kv) for p in self.train])
        for p in self.train:
            p["weaknesses"] = [review_weaknesses(r) for r in p["reviews"]]
        # Global category frequency over training reviews: only used to fill when
        # the neighbors raise fewer than 5 distinct categories.
        self.category_prior = category_prior(w for p in self.train for w in p["weaknesses"])

    def neighbors(self, text):
        q = embed(text, self.kv)
        sims = self.train_emb @ q
        idx = np.argsort(-sims)[: self.k]
        return [(self.train[i], float(sims[i])) for i in idx]

    def review(self, title, abstract, n_changes=5):
        nbrs = self.neighbors(f"{title}. {abstract}")
        weights = np.array([max(s, 0.0) for _, s in nbrs])
        if weights.sum() <= 0:
            weights = np.ones(len(nbrs))
        weights = weights / weights.sum()

        rating = float(sum(w * p["rating"] for w, (p, _) in zip(weights, nbrs)))

        # Score = similarity-weighted fraction of each neighbor's reviewers raising the category.
        scores, evidence = Counter(), {}
        for w, (p, _) in zip(weights, nbrs):
            n_rev = len(p["weaknesses"])
            for rev in p["weaknesses"]:
                for cat, sents in rev.items():
                    scores[cat] += w / n_rev
                    # nbrs is sorted by similarity, so the first quote seen is from the closest neighbor
                    evidence.setdefault(cat, (sents[0], p["id"]))

        order = {c: i for i, c in enumerate(self.category_prior)}
        ranked = sorted(scores, key=lambda c: (-scores[c], order[c]))
        ranked += [c for c in self.category_prior if c not in ranked]
        changes = []
        for cat in ranked[:n_changes]:
            quote, src = evidence.get(cat, (None, None))
            changes.append({"category": cat, "change": ACTIONS[cat], "score": round(scores[cat], 4),
                            "evidence": quote, "evidence_paper_id": src})

        return {
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
        print(f"    {i}. [{c['category']}, score={c['score']:.3f}] {c['change']}")
        if c["evidence"]:
            print(f"       e.g. reviewer of paper {c['evidence_paper_id']}: \"{c['evidence'][:200]}\"")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--k", type=int, default=5, help="number of nearest training papers")
    ap.add_argument("--w2v", default="word2vec-google-news-300",
                    help="gensim-downloader model name, or path to a .bin/.txt/.kv word2vec file")
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                    help="dir with {train,dev,test}/reviews/*.json (e.g. PeerRead/data/iclr_2017)")
    ap.add_argument("--show", type=int, default=3, help="number of example reviews to print")
    ap.add_argument("--out", default=None, help="output JSONL path (default: baseline/nonml_iclr2017_<split>.jsonl)")
    args = ap.parse_args()

    train = load_split(args.data_dir, "train")
    test = load_split(args.data_dir, args.split)
    print(f"Loaded {len(train)} train / {len(test)} {args.split} papers "
          f"({sum(bool(p['reviews']) for p in train)} train papers with rated reviews)")

    kv = load_word2vec(args.w2v)
    reviewer = NonMLReviewer(train, kv, k=args.k)

    out_path = args.out or os.path.join(HERE, f"nonml_iclr2017_{args.split}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for i, p in enumerate(test):
            r = {"id": p["id"], "title": p["title"], **reviewer.review(p["title"], p["abstract"])}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i < args.show:
                print_review(r)
    print("=" * 100)
    print(f"Wrote {len(test)} reviews to {out_path}")
    print(f"Evaluate with: python evaluation/eval.py {out_path}")


if __name__ == "__main__":
    main()
