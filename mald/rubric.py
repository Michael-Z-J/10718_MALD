"""common rubric: map the review form fields of each venue to four axes in [0, 1].

note content --> leading int of each field --> fields per axis --> min-max
scale with the range of that venue-year in the corpus.

axes: soundness, presentation, contribution, overall. an axis that a venue
does not rate stays None (no imputation).
"""
import re

AXES = ("soundness", "presentation", "contribution", "overall")

# venue field name(s) per axis. a list means the mean of the fields.
FIELDS = {
    "soundness": ["correctness", "soundness", "technical_quality"],
    "presentation": ["presentation", "clarity_of_presentation"],
    "contribution": [
        ["technical_novelty_and_significance", "empirical_novelty_and_significance"],
        "contribution",
    ],
    "overall": ["recommendation", "rating", "overall_rating"],
}


def num(v):
    """leading int of a form value such as '8: accept, good paper', else None."""
    if isinstance(v, (int, float)):
        return float(v)
    m = re.match(r"\s*(-?\d+)", str(v or ""))
    return float(m.group(1)) if m else None


def raw(content):
    """raw axis values of one review note, before scaling."""
    out = {}
    for ax, opts in FIELDS.items():
        out[ax] = None
        for f in opts:
            names = f if isinstance(f, list) else [f]
            vals = [num(content.get(n)) for n in names if n in content]
            vals = [x for x in vals if x is not None]
            if vals:
                out[ax] = sum(vals) / len(vals)
                break
    return out


def ranges(rows):
    """per (venue, axis) observed (lo, hi) over all reviews of that venue-year.

    rows: iterable of (venue, raw_dict).
    """
    rng = {}
    for v, r in rows:
        for ax, x in r.items():
            if x is None:
                continue
            lo, hi = rng.get((v, ax), (x, x))
            rng[(v, ax)] = (min(lo, x), max(hi, x))
    return rng


def scale(v, r, rng):
    """scale raw axis values to [0, 1] with the venue-year range."""
    out = {}
    for ax, x in r.items():
        lo, hi = rng.get((v, ax), (None, None))
        out[ax] = None if x is None or hi is None or hi == lo else (x - lo) / (hi - lo)
    return out
