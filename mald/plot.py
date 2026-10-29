"""summary.json --> curve figure for the report (pdf).

usage: python -m mald.plot <split> <out.pdf> <name=label> [...]
       python -m mald.plot judge <agg.json> <out.pdf> <name=label> [...]
the judge form draws the judge-matched curves against c and, in a second
file, the minilm-matched curves on the same papers (mean over tau).
grid of panels: rows = metrics, columns = (gt grouping threshold c,
matching threshold tau). one line per system, pooled population value.
each system has a fixed color and its own marker. the left panels label the
line ends and the right panels have a legend.
"""
import json, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mald.paths import OUT
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
METRICS = [("ndcg_cons", "nDCG (consensus gain)"), ("recall_w", "consensus-weighted recall"),
           ("recall_multi", "recall of concerns raised by $\\geq$2 reviewers"), ("precision", "precision")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


def main(split, out, *pairs):
    d = json.load(open(f"{OUT}/{split}/summary.json"))
    s, cg, tg = d["systems"], d["cgrid"], d["taus"]
    names = [(a, b.replace("_", " ")) for a, b in (p.split("=", 1) for p in pairs)]
    names = [(n, l) for n, l in names if n in s]
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(len(METRICS), 2, figsize=(6.6, 2.0 * len(METRICS)), sharex="col")
    for i, (m, title) in enumerate(METRICS):
        for j, (key, xs, xl) in enumerate((("curve_c", cg, "gt grouping threshold $c$"),
                                           ("curve_tau", tg, "matching threshold $\\tau$"))):
            a = ax[i, j]
            a.grid(axis="y", color=GRID, linewidth=0.6)
            for k, (n, lab) in enumerate(names):
                ys = s[n][key][m]
                a.plot(xs, ys, color=COLORS[k], linewidth=2, marker=MARKERS[k], markersize=4, label=lab)
                if j == 0 and ys[-1] == ys[-1]:
                    a.annotate(lab, (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", fontsize=6.5,
                               color=INK, va="center")
            a.set_title(title if j == 0 else "", fontsize=8, color=INK, loc="left")
            if i == len(METRICS) - 1:
                a.set_xlabel(xl)
            a.set_xlim(xs[0], xs[-1] + (xs[-1] - xs[0]) * 0.22)
    ax[0, 1].legend(frameon=False, fontsize=6.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(out)


def judge(agg, out, *pairs):
    """two figures: judge-matched curves (out) and minilm-matched curves on the
    same papers (out with an _emb suffix), with equal y limits per metric."""
    d = json.load(open(agg))
    cg = d["_cgrid"]
    names = [(a, b.replace("_", " ")) for a, b in (p.split("=", 1) for p in pairs)]
    names = [(n, l) for n, l in names if n in d]
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    ys = {m: [y for n, _ in names for k in ("curve_c", "emb_curve_c") for y in d[n][k][m] if y == y] for m, _ in METRICS}
    lim = {m: (0, max(v) * 1.08) for m, v in ys.items()}
    for key, path in (("curve_c", out), ("emb_curve_c", out.replace(".pdf", "_emb.pdf"))):
        fig, ax = plt.subplots(2, 2, figsize=(6.6, 4.2), sharex=True)
        for a, (m, title) in zip(ax.flat, METRICS):
            a.grid(axis="y", color=GRID, linewidth=0.6)
            for k, (n, lab) in enumerate(names):
                a.plot(cg, d[n][key][m], color=COLORS[k], linewidth=2, marker=MARKERS[k], markersize=4, label=lab)
            a.set_title(title, fontsize=8, color=INK, loc="left")
            a.set_ylim(*lim[m])
        for a in ax[1]:
            a.set_xlabel("gt grouping threshold $c$")
        h, l = ax[0, 0].get_legend_handles_labels()
        fig.legend(h, l, frameon=False, fontsize=6.5, loc="lower center", ncol=4)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(path)
        plt.close(fig)


if __name__ == "__main__":
    if sys.argv[1] == "judge":
        judge(*sys.argv[2:])
    else:
        main(*sys.argv[1:])
