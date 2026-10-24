"""
============================================================================
 rq4_analysis.py, RQ4: Scanner Accuracy Against Manual Ground Truth
============================================================================
 Question: How accurately does the automated scanner flag risky servers?
 Approach: Reads the scored output of labeling/02_score_scanner.py and
           assembles the paper-facing accuracy section: per-class precision
           with Wilson intervals, pooled precision, the miss rate from the
           unflagged stratum, and prevalence corrected for measured
           precision. Fails clearly if labelling is not yet complete.
 Output  : analysis/rq4_scanner_accuracy/outputs/rq4_results.{csv,json}
           rq4_apa_report.txt + figures/
============================================================================
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from setup import config                                              # noqa: E402
from utils.io_helpers import load_master, save_csv, save_json, save_txt  # noqa: E402
from utils import stats as st                                         # noqa: E402
from utils.reporting import ReportBuilder                             # noqa: E402

RQ = "rq4"
OUT, FIG = config.rq_output_dir(RQ), config.rq_figure_dir(RQ)
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300})
BLUE, RED, GREY = "#2E5E8C", "#B0413E", "#7A7A7A"


def main():
    print("=" * 76)
    print(" RQ4 — SCANNER ACCURACY")
    print("=" * 76)
    acc_json = config.LABELING_DIR / "scanner_accuracy.json"
    if not acc_json.exists():
        sys.exit("[FAIL] scanner_accuracy.json missing.\n"
                 "  Complete the labelling sheet, then run\n"
                 "  python labeling/02_score_scanner.py")
    A = json.loads(acc_json.read_text())
    per = pd.DataFrame(A["per_class"])
    print(f"  labelled rows     : {A['n_labelled']:,}")
    print(f"  pooled precision  : {A['pooled_precision']:.3f} "
          f"[{A['pooled_precision_ci'][0]:.3f}, "
          f"{A['pooled_precision_ci'][1]:.3f}]")

    rb = ReportBuilder(RQ)
    rb.population(rows_labelled=A["n_labelled"],
                  flagged_stratum=int(per["n_labelled"].sum()),
                  unflagged_stratum=A["unflagged_sampled"],
                  classes_scored=len(per))
    rb.text("Findings were sampled stratified by threat class, at most one",
            "finding per server, and interleaved so that classes were judged",
            "in mixed order. A second stratum of files with no finding was",
            "inspected to estimate how much the scanner misses.")

    rb.classification_result(per.to_dict("records"),
                             "Per-class accuracy against manual labels")
    rb.section("Pooled precision")
    lo, hi = A["pooled_precision_ci"]
    rb.text(f"Pooled precision across all labelled findings: "
            f"{A['pooled_precision']:.3f}, 95% CI [{lo:.3f}, {hi:.3f}].")

    rb.section("Missed findings")
    n_unf, n_miss = A["unflagged_sampled"], A["unflagged_missed"]
    mlo, mhi = A["miss_rate_ci"]
    rate = n_miss / n_unf if n_unf else float("nan")
    rb.text(f"Of {n_unf:,} files with no finding, {n_miss:,} were judged to "
            f"contain an issue: an estimated miss rate of {rate:.3f}, "
            f"95% CI [{mlo:.3f}, {mhi:.3f}].",
            "Population recall is not identifiable because the true number of",
            "issues in unscanned code is unknown; this rate bounds how much",
            "the reported prevalences understate the ecosystem.")

    corr = pd.DataFrame(A.get("corrected_prevalence", []))
    if len(corr):
        show = corr.copy()
        for c in ["raw_prevalence", "corrected_prevalence",
                  "corrected_low", "corrected_high"]:
            show[c] = (show[c] * 100).round(2)
        rb.dataframe_table(show, "Precision-corrected prevalence (%)")
        rb.text("", "Corrected prevalence multiplies the observed rate by the",
                "measured precision for that class, giving a conservative",
                "estimate. Reported intervals combine the precision interval",
                "with the observed count.")

    if A.get("kappa"):
        k = A["kappa"]
        rb.section("Inter-rater agreement")
        rb.text(f"Shared rows: {k['n']:,}. Observed agreement "
                f"{k['observed_agreement']:.3f}. Cohen's kappa "
                f"{k['kappa']:.3f}.")

    rb.section("Interpretation")
    worst = per.loc[per["precision"].idxmin()]
    best = per.loc[per["precision"].idxmax()]
    rb.text(f"Precision was highest for {best['class']} "
            f"({best['precision']:.2f}) and lowest for {worst['class']} "
            f"({worst['precision']:.2f}). Classes defined by a narrow",
            "syntactic signature are detected more reliably than classes",
            "that depend on how a value is later used, which is the expected",
            "ordering for pattern-based analysis and sets the boundary for",
            "what this method can claim.")
    rb.detection_caveat()

    # ---- figure: precision with CIs ----------------------------------------
    plot = per.sort_values("precision")
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    for i, (_, r) in enumerate(plot.iterrows()):
        c = BLUE if r["precision"] >= .7 else (
            GREY if r["precision"] >= .5 else RED)
        ax.plot([r["precision_ci_low"], r["precision_ci_high"]], [i, i],
                color=c, lw=1.3)
        for x in (r["precision_ci_low"], r["precision_ci_high"]):
            ax.plot([x, x], [i - .13, i + .13], color=c, lw=1.0)
        ax.plot(r["precision"], i, "o", color=c, ms=4.2,
                mec="white", mew=.5, zorder=5)
        ax.annotate(f"{r['precision']:.2f}", (r["precision_ci_high"], i),
                    xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=6.4, color=c)
    ax.axvline(A["pooled_precision"], color="#3A3A3A", lw=.8, ls=(0, (4, 2)))
    ax.set_yticks(range(len(plot)))
    ax.set_yticklabels([c.split("_", 1)[1].replace("_", " ")
                        for c in plot["class"]])
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("precision, 95% Wilson CI (dashed: pooled)")
    ax.grid(axis="x", color="#DDDDDD", lw=.45); ax.set_axisbelow(True)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "rq4_precision.pdf"); plt.close(fig)

    if len(corr):
        fig, ax = plt.subplots(figsize=(3.45, 2.1))
        cc = corr.sort_values("raw_prevalence")
        y = np.arange(len(cc))
        ax.barh(y - .2, cc["raw_prevalence"] * 100, height=.38,
                color=GREY, label="observed")
        ax.barh(y + .2, cc["corrected_prevalence"] * 100, height=.38,
                color=BLUE, label="precision-corrected")
        ax.set_yticks(y)
        ax.set_yticklabels([c.split("_", 1)[1].replace("_", " ")
                            for c in cc["class"]])
        ax.set_xlabel("servers affected (%)")
        ax.legend(frameon=False, fontsize=6.4)
        ax.grid(axis="x", color="#DDDDDD", lw=.45); ax.set_axisbelow(True)
        fig.tight_layout(); fig.savefig(FIG / "rq4_corrected.pdf")
        plt.close(fig)
    print(f"  [SAVED] figures -> {FIG}")

    save_csv(per, OUT / "rq4_results.csv", "rq4 accuracy")
    save_json(A, OUT / "rq4_results.json", "rq4 results")
    save_txt(rb.render(), OUT / "rq4_apa_report.txt", "rq4 APA report")
    print("  RQ4 COMPLETE")


if __name__ == "__main__":
    main()
