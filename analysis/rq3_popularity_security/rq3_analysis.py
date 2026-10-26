"""
============================================================================
 rq3_analysis.py, RQ3: Does Popularity Predict Security Posture?
============================================================================
 Question: Does a server's popularity predict its security posture?
 Approach: Rank-based association (Spearman) between popularity and finding
           counts, because stars span several orders of magnitude; prevalence
           of high-severity findings by tie-aware star bins (half the servers
           have zero stars, so rank deciles would split tied servers by row
           order and are not interpreted); logistic
           regression with popularity, maintenance, age, and size as
           predictors, reporting odds ratios; BH correction across the
           association tests.
 Output  : analysis/rq3_popularity_security/outputs/rq3_results.{csv,json}
           rq3_apa_report.txt + figures/
============================================================================
"""

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

RQ = "rq3"
OUT, FIG = config.rq_output_dir(RQ), config.rq_figure_dir(RQ)
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300})
BLUE, RED, GREY = "#2E5E8C", "#B0413E", "#7A7A7A"
PAPER_BLUE, PAPER_GREEN = "#4C72B0", "#55A868"   # palette of the paper figures


def main():
    print("=" * 76)
    print(" RQ3 — POPULARITY AND SECURITY POSTURE")
    print("=" * 76)
    df = load_master(config.FILE_MASTER)
    d = df[(df["has_code"].astype(bool)) & df["stars"].notna()].copy()
    print(f"  servers with code and popularity data : {len(d):,}")
    print(f"  stars: median {d['stars'].median():.0f}, "
          f"max {d['stars'].max():.0f}, zero-star "
          f"{(d['stars'] == 0).mean():.0%}")

    rb = ReportBuilder(RQ)
    rb.population(servers_scanned=int(df["has_code"].sum()),
                  with_popularity_data=len(d),
                  median_stars=int(d["stars"].median()),
                  max_stars=int(d["stars"].max()),
                  zero_star_servers=int((d["stars"] == 0).sum()))
    rb.text("Star counts span several orders of magnitude and are heavily",
            "zero-inflated, so associations use rank-based methods and the",
            "regression uses log(1 + stars) rather than raw counts.")

    # ---- associations -------------------------------------------------------
    assoc = []
    pairs = [("log_stars", "n_findings_total", "popularity vs findings"),
             ("log_stars", "severity_score", "popularity vs severity score"),
             ("days_since_push", "n_findings_total",
              "days since last push vs findings"),
             ("repo_age_days", "n_findings_total", "repository age vs findings")]
    for x, y, label in pairs:
        if x not in d or y not in d:
            continue
        r = st.spearman(d[x], d[y])
        r["pair"] = label
        assoc.append(r)
        print(f"    {label:<34s} rho={r['rho']:+.3f} {st.format_p(r['p'])}")
    if assoc:
        p_adj = st.adjust_pvalues([a["p"] for a in assoc])
        for a, pa in zip(assoc, p_adj):
            a["p_adj"] = pa
            rb.spearman_result(a, a["pair"])

    # ---- prevalence by star bin ----------------------------------------------
    # Half the scanned servers have zero stars, so rank deciles 1-5 would all
    # be zero-star servers separated only by row order. Bins instead close at
    # the upper star count of each rank decile, which keeps tied servers
    # together: 0 | 1 | 2 | 3-6 | 7-28 | 29+ on the August 2026 snapshot.
    dec_rank = pd.qcut(d["stars"].rank(method="first"), 10, labels=False)
    edges = sorted(set(d.groupby(dec_rank)["stars"].max().astype(int)))[:-1]
    lowers = [0] + [e + 1 for e in edges]
    uppers = edges + [None]
    rows = []
    for lo_s, hi_s in zip(lowers, uppers):
        if hi_s is None:
            sub, label = d[d["stars"] >= lo_s], f"{lo_s}+"
        else:
            sub = d[(d["stars"] >= lo_s) & (d["stars"] <= hi_s)]
            label = str(lo_s) if lo_s == hi_s else f"{lo_s}-{hi_s}"
        k = int(sub["has_high_severity"].fillna(0).sum())
        p = st.prevalence(k, len(sub), label)
        p["min_stars"], p["max_stars"] = float(sub["stars"].min()), float(sub["stars"].max())
        rows.append(p)
    bin_df = pd.DataFrame(rows)
    rb.prevalence_table(bin_df.to_dict("records"),
                        "High-severity prevalence by star bin (tie-aware; "
                        "bins close at rank-decile boundaries)")
    for r in rows:
        print(f"    stars {r['group']:<6s} n={r['n']:>6,}  {r['prevalence']:>7.2%}")

    # most popular bin vs zero-star servers
    lo_b, hi_b = bin_df.iloc[0], bin_df.iloc[-1]
    con = st.two_proportion_test(int(hi_b["k"]), int(hi_b["n"]),
                                 int(lo_b["k"]), int(lo_b["n"]),
                                 f"{hi_b['group']} stars", "zero stars")
    rb.chi_square_result(con, "High-severity findings, most popular bin vs "
                              "zero-star servers")
    print(f"    {hi_b['group']} vs 0 stars OR="
          f"{con.get('odds_ratio', float('nan')):.2f} "
          f"[{con.get('or_ci_low', float('nan')):.2f}, "
          f"{con.get('or_ci_high', float('nan')):.2f}] {st.format_p(con['p'])}")

    # ---- logistic regression -------------------------------------------------
    d["log_files"] = np.log1p(d["n_files"].fillna(0))
    d["log_age"] = np.log1p(d["repo_age_days"].fillna(0))
    preds = [c for c in ["log_stars", "log_files", "log_age", "is_maintained",
                         "is_archived", "has_license"] if c in d]
    model_df = d[d["has_high_severity"].notna()].copy()
    model_df["outcome"] = model_df["has_high_severity"].astype(int)
    lr = st.logistic_regression(model_df, "outcome", preds)
    if not lr.empty:
        rb.logistic_result(lr, "any high-severity finding")
        for _, r in lr.iterrows():
            if r["term"] != "const":
                print(f"    {r['term']:<16s} OR={r['odds_ratio']:.3f} "
                      f"[{r['or_ci_low']:.3f}, {r['or_ci_high']:.3f}] "
                      f"{st.format_p(r['p'])}")
        rb.text("", "Odds ratios are per unit of the log-transformed",
                "predictor, so a value above 1 means the outcome becomes more",
                "likely as the predictor increases. Size of the fetched code",
                "base is included because a larger code base offers more",
                "opportunities for any pattern to appear.")

    # ---- maintenance contrast -------------------------------------------------
    if "is_maintained" in d:
        m1 = d[d["is_maintained"] == 1]
        m0 = d[d["is_maintained"] == 0]
        if len(m1) > 30 and len(m0) > 30:
            mt = st.two_proportion_test(
                int(m1["has_high_severity"].fillna(0).sum()), len(m1),
                int(m0["has_high_severity"].fillna(0).sum()), len(m0),
                "maintained", "stale")
            rb.chi_square_result(mt, "High-severity findings, maintained "
                                     "(pushed within 180 days) vs stale")

    rb.methods_note()
    rb.detection_caveat()

    # ---- figures ---------------------------------------------------------------
    with plt.rc_context({"font.family": "serif",
                         "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                         "font.size": 8, "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(3.45, 2.05))
        x = np.arange(len(bin_df))
        top = len(bin_df) - 1
        cols = [PAPER_GREEN if i == top else PAPER_BLUE for i in x]
        y = bin_df["prevalence"] * 100
        err = [y - bin_df["ci_low"] * 100, bin_df["ci_high"] * 100 - y]
        ax.bar(x, y, width=.62, color=cols, zorder=2)
        ax.errorbar(x, y, yerr=err, fmt="none", ecolor="#222222",
                    elinewidth=.7, capsize=2, zorder=3)
        for i, (v, hi_) in enumerate(zip(y, bin_df["ci_high"] * 100)):
            ax.text(i, hi_ + .5, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g.replace('-', chr(8211))}\nn={n:,}"
                            for g, n in zip(bin_df["group"], bin_df["n"])],
                           fontsize=6.6)
        ax.set_xlabel("GitHub stars (tie-aware bins)")
        ax.set_ylabel("high-severity servers (%)")
        ax.set_ylim(0, max(bin_df["ci_high"] * 100) + 3.2)
        ax.grid(axis="y", color="#CCCCCC", lw=.45, ls=(0, (2, 2)))
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIG / "rq3_stars.pdf")
        plt.close(fig)

    if not lr.empty:
        terms = lr[lr["term"] != "const"]
        fig, ax = plt.subplots(figsize=(3.45, 2.0))
        for i, (_, r) in enumerate(terms.iterrows()):
            sig = r["p"] < config.ALPHA
            c = RED if sig else GREY
            ax.plot([r["or_ci_low"], r["or_ci_high"]], [i, i], color=c, lw=1.3)
            ax.plot(r["odds_ratio"], i, "s", color=c, ms=4,
                    mec="white", mew=.4, zorder=5)
        ax.axvline(1, color="#3A3A3A", lw=.8, ls=(0, (4, 2)))
        ax.set_yticks(range(len(terms)))
        ax.set_yticklabels([t.replace("_", " ") for t in terms["term"]])
        ax.set_xscale("log")
        ax.set_xlabel("odds ratio for a high-severity finding")
        ax.grid(axis="x", color="#DDDDDD", lw=.45); ax.set_axisbelow(True)
        ax.grid(axis="y", visible=False)
        fig.tight_layout(); fig.savefig(FIG / "rq3_logistic.pdf"); plt.close(fig)
    print(f"  [SAVED] figures -> {FIG}")

    save_csv(bin_df, OUT / "rq3_results.csv", "rq3 star bins")
    save_json({"n": len(d), "associations": assoc,
               "star_bins": bin_df.to_dict("records"),
               "top_bin_vs_zero_stars": con,
               "logistic": lr.to_dict("records") if not lr.empty else [],
               "logistic_n": int(lr.attrs.get("n", 0)) if not lr.empty else 0},
              OUT / "rq3_results.json", "rq3 results")
    save_txt(rb.render(), OUT / "rq3_apa_report.txt", "rq3 APA report")
    print("  RQ3 COMPLETE")


if __name__ == "__main__":
    main()
