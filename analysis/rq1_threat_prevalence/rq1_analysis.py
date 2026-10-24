"""
============================================================================
 rq1_analysis.py, RQ1: Threat-Class Prevalence Across the Population
============================================================================
 Question: How prevalent is each MCP threat class across the public server
           population?
 Approach: Prevalence with Wilson intervals per class over the scan
           population; severity rollup; co-occurrence of classes; prevalence
           by ecosystem stratum (package registry, transport) with
           chi-square and Cramer's V, BH-corrected across strata tests.
           Zero-count classes are reported with their Wilson upper bound
           rather than dropped.
 Output  : analysis/rq1_threat_prevalence/outputs/rq1_results.{csv,json}
           rq1_apa_report.txt + figures/
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

RQ = "rq1"
OUT, FIG = config.rq_output_dir(RQ), config.rq_figure_dir(RQ)
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300})
BLUE, RED, GREY = "#2E5E8C", "#B0413E", "#7A7A7A"


def main():
    print("=" * 76)
    print(" RQ1 — THREAT-CLASS PREVALENCE")
    print("=" * 76)
    df = load_master(config.FILE_MASTER)
    scanned = df[df["has_code"].astype(bool)].copy()
    n = len(scanned)
    print(f"  scan population : {n:,} servers "
          f"({n/len(df):.1%} of the registry)")

    rb = ReportBuilder(RQ)
    rb.population(servers_in_registry=len(df),
                  servers_with_repository_url=int(df["has_repo"].sum()),
                  servers_scanned=n,
                  servers_with_any_finding=int(scanned["any_finding"].sum()))
    rb.text("Prevalence is computed over the scan population: servers whose",
            "source code was retrieved. Servers without retrievable code are",
            "excluded from denominators rather than assumed clean.")

    # ---- per-class prevalence ---------------------------------------------
    rows = []
    for cls, spec in config.THREAT_CLASSES.items():
        col = f"has_{cls}"
        k = int(scanned[col].fillna(0).sum()) if col in scanned else 0
        p = st.prevalence(k, n, cls)
        p["severity"] = spec["severity"]
        p["name"] = spec["name"]
        rows.append(p)
        print(f"    {cls:<30s} {k:>6,}  {p['prevalence']:>7.2%}  "
              f"[{p['ci_low']:.4f}, {p['ci_high']:.4f}]")
    prev = pd.DataFrame(rows).sort_values("prevalence", ascending=False)
    rb.prevalence_table(prev.to_dict("records"),
                        "Threat-class prevalence (scan population)")

    zero = prev[prev["k"] == 0]
    if len(zero):
        rb.section("Zero-prevalence classes")
        for _, z in zero.iterrows():
            rb.text(f"{z['group']}: 0 of {n:,} servers. The 95% Wilson upper "
                    f"bound is {z['ci_high']*100:.3f}%, so prevalence above "
                    f"that is inconsistent with the data. Absence of "
                    f"detection is not proof of absence: pattern-based "
                    f"detection cannot observe obfuscated variants.")

    # ---- severity rollup ---------------------------------------------------
    sev_rows = []
    for sev in ["high", "medium", "low"]:
        cols = [f"has_{c}" for c, s in config.THREAT_CLASSES.items()
                if s["severity"] == sev and f"has_{c}" in scanned]
        if not cols:
            continue
        k = int((scanned[cols].fillna(0).sum(axis=1) > 0).sum())
        sev_rows.append(st.prevalence(k, n, f"any {sev}-severity class"))
    k_any = int((scanned["any_finding"].fillna(0) > 0).sum())
    sev_rows.append(st.prevalence(k_any, n, "any class"))
    rb.prevalence_table(sev_rows, "Severity rollup")

    # ---- co-occurrence ------------------------------------------------------
    has_cols = [f"has_{c}" for c in config.THREAT_CLASSES
                if f"has_{c}" in scanned]
    counts = scanned[has_cols].fillna(0).sum(axis=1)
    co = (counts.value_counts().sort_index()
          .rename_axis("classes_present").reset_index(name="servers"))
    co["share"] = co["servers"] / n
    rb.dataframe_table(co, "Number of distinct threat classes per server")
    print(f"  servers with >1 class : {int((counts > 1).sum()):,} "
          f"({(counts > 1).mean():.2%})")

    # ---- prevalence by ecosystem stratum ------------------------------------
    strata_tests = []
    for stratum, label in [("primary_package", "package registry"),
                           ("primary_transport", "transport")]:
        sub = scanned[scanned[stratum].notna()].copy()
        top = sub[stratum].value_counts()
        keep = top[top >= 200].index.tolist()
        sub = sub[sub[stratum].isin(keep)]
        if sub[stratum].nunique() < 2:
            continue
        tab = pd.crosstab(sub[stratum],
                          sub["has_high_severity"].fillna(0).astype(int))
        if tab.shape[1] < 2:
            continue
        res = st.chi_square(tab.values)
        res.update({"stratum": stratum, "levels": ", ".join(map(str, keep))})
        strata_tests.append(res)
        breakdown = (sub.groupby(stratum)["has_high_severity"]
                     .agg(["sum", "count"]).reset_index())
        breakdown.columns = [stratum, "k", "n"]
        breakdown["prevalence"] = breakdown["k"] / breakdown["n"]
        rb.dataframe_table(breakdown.round(4),
                           f"High-severity prevalence by {label}")
        print(f"  by {label:<18s} chi2={res['chi2']:.1f} "
              f"{st.format_p(res['p'])} V={res['cramers_v']:.3f}")
    if strata_tests:
        p_adj = st.adjust_pvalues([r["p"] for r in strata_tests])
        for r, pa in zip(strata_tests, p_adj):
            rb.chi_square_result(
                r, f"Any high-severity finding by {r['stratum']}", p_adj=pa)

    rb.methods_note()
    rb.detection_caveat()

    # ---- figures -------------------------------------------------------------
    plot = prev[prev["k"] > 0].sort_values("prevalence")
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    for i, (_, r) in enumerate(plot.iterrows()):
        c = RED if r["severity"] == "high" else (
            BLUE if r["severity"] == "medium" else GREY)
        ax.plot([r["ci_low"] * 100, r["ci_high"] * 100], [i, i], color=c, lw=1.3)
        for x in (r["ci_low"], r["ci_high"]):
            ax.plot([x * 100, x * 100], [i - .13, i + .13], color=c, lw=1.0)
        ax.plot(r["prevalence"] * 100, i, "o", color=c, ms=4.2,
                mec="white", mew=.5, zorder=5)
        ax.annotate(f"{r['prevalence']*100:.2f}%",
                    (r["ci_high"] * 100, i), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=6.4,
                    color=c)
    ax.set_yticks(range(len(plot)))
    ax.set_yticklabels([g.split("_", 1)[1].replace("_", " ")
                        for g in plot["group"]])
    ax.set_xlabel("servers affected (%), 95% Wilson CI")
    ax.grid(axis="x", color="#DDDDDD", lw=.45)
    ax.set_axisbelow(True)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(plot["ci_high"]) * 100 * 1.35)
    fig.tight_layout()
    fig.savefig(FIG / "rq1_prevalence.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.45, 1.9))
    ax.bar(co["classes_present"].astype(int), co["share"] * 100, color=BLUE)
    ax.set_xlabel("distinct threat classes present")
    ax.set_ylabel("servers (%)")
    ax.grid(axis="y", color="#DDDDDD", lw=.45)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "rq1_cooccurrence.pdf")
    plt.close(fig)
    print(f"  [SAVED] figures -> {FIG}")

    save_csv(prev, OUT / "rq1_results.csv", "rq1 prevalence")
    save_json({"n_scanned": n, "prevalence": prev.to_dict("records"),
               "severity_rollup": sev_rows,
               "co_occurrence": co.to_dict("records"),
               "stratum_tests": strata_tests},
              OUT / "rq1_results.json", "rq1 results")
    save_txt(rb.render(), OUT / "rq1_apa_report.txt", "rq1 APA report")
    print("  RQ1 COMPLETE")


if __name__ == "__main__":
    main()
