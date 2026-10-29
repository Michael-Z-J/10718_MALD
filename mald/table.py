"""summary.json --> latex tables for the report.

usage: python -m mald.table <split> <name=label> [...] > table.tex
"""
import json, sys

from mald.paths import OUT
META = [("recall_w", "rec$_w$"), ("recall_multi", "rec$_{\\ge2}$"), ("ndcg_cons", "nDCG$_c$"), ("ndcg_def", "nDCG$_d$"),
        ("precision", "prec"), ("recall_at_5", "rec@5"), ("resolved_recall", "res.rec"), ("resolved_flagged", "res.flag"),
        ("flag_lift", "lift"), ("p5_legacy", "p@5$^\\dagger$"), ("n_issues", "\\#iss"), ("j_ndcg_cons", "nDCG$_c^J$"), ("j_recall_w", "rec$_w^J$"), ("j_resolved_recall", "res.rec$^J$")]
ALLOC = [("alloc_best", "best"), ("alloc_rand", "random"), ("alloc_worst", "worst"), ("panel_union", "any bot")]
PANEL = [("alloc_best", "cov$_{best}$"), ("alloc_rand", "cov$_{rand}$"), ("alloc_worst", "cov$_{worst}$"),
         ("panel_union", "cov$_\\cup$"), ("score_mae_best", "mae$_{best}$"), ("panel_overall_spearman", "$\\rho_{ov}$"),
         ("panel_soundness_spearman", "$\\rho_{snd}$"), ("panel_contribution_spearman", "$\\rho_{ctr}$"),
         ("panel_overall_mae", "mae$_{ov}$"), ("panel_accept_auc", "AUC"), ("decision_acc", "dec.acc"),
         ("disagreement_spearman", "$\\rho_{sd}$")]


def fmt(x):
    return "n/a" if x is None or x != x else (f"{x:.0f}" if abs(x) >= 2 else f"{x:.3f}")


def tab(summ, names, cols, cap):
    head = " & ".join(["system"] + [c for _, c in cols])
    rows = [" & ".join([lab] + [fmt(summ[n].get(k)) for k, _ in cols]) + r" \\" for n, lab in names]
    return "\n".join([(r"\begin{table}[H]\centering\scriptsize\setlength{\tabcolsep}{3pt}" if len(cols) > 6 else r"\begin{table}[H]\centering\small"),
                      *([r"\resizebox{\linewidth}{!}{%"] if len(cols) > 6 else []), r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule", head + r" \\", r"\midrule",
                      *rows, r"\bottomrule", r"\end{tabular}" + ("}" if len(cols) > 6 else ""), rf"\caption{{{cap}}}", r"\end{table}"])


COV = [("recall_union", "rec"), ("recall_w", "rec$_w$"), ("recall_multi", "rec$_{\\ge2}$"), ("recall_at_5", "rec@5"), ("recall_at_10", "rec@10"),
       ("ndcg_cons", "nDCG$_c$"), ("ndcg_def", "nDCG$_d$"), ("precision", "prec"), ("p5_legacy", "p@5$^\\dagger$"), ("n_issues", "\\#iss")]
FLAG = [("resolved_recall", "res.rec"), ("resolved_flagged", "res.flag"), ("flag_lift", "lift"),
        ("j_recall_w", "rec$_w^J$"), ("j_recall_multi", "rec$_{\\ge2}^J$"), ("j_ndcg_cons", "nDCG$_c^J$"), ("j_precision", "prec$^J$"), ("j_resolved_recall", "res.rec$^J$")]


def main(split, *pairs):
    s = json.load(open(f"{OUT}/{split}/summary.json"))["systems"]
    names = [(a, b.replace("_", " ")) for a, b in (p.split("=", 1) for p in pairs)]
    names = [(n, l) for n, l in names if n in s]
    n = max(s[k].get("n_papers", 0) for k, _ in names)
    who = "Test" if split == "test" else "Validation"
    print(tab(s, names, COV, f"{who} set ({n} papers): coverage and ranking of the meta-review. rec: share of real concerns stated (unweighted); rec$_w$: consensus-weighted recall; "
              "rec$_{{\\ge2}}$: recall of concerns raised by two or more reviewers; nDCG$_c$, nDCG$_d$: consensus and deficit gain; "
              "p@5$^\\dagger$: precision@5 as defined in the previous report; \\#iss: issues per meta-review. "
              "Population values pooled over all papers, grouping thresholds $c\\in[0.30,0.80]$ and matching thresholds $\\tau\\in[0.40,0.80]$ (curves in the figure)."))
    flag = FLAG if split == "test" else FLAG[:3]
    print(tab(s, names, flag, f"{who} set: doability and relevance flags"
              + (", and the same coverage metrics with the LLM judge deciding matches (superscript $J$)" if split == "test" else "")
              + ". res.rec: recall of concerns the reviewer called resolved after the rebuttal; res.flag: share of those found concerns flagged relevant and doable; "
              "lift: match rate of issues flagged relevant minus that of issues flagged not relevant."))
    print(tab(s, names, ALLOC, f"{who} set: share of a real reviewer's concerns stated by the simulated reviewer allocated to them, "
              "under the best, random and worst assignment, and the share stated by any simulated reviewer."))


if __name__ == "__main__":
    main(*sys.argv[1:])
