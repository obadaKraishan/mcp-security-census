"""
============================================================================
 rq5_analysis.py, RQ5: Ambient Authority and Token Passthrough
============================================================================
 Question: What share of servers expose ambient authority or pass caller
           tokens through to third parties?
 Approach: Prevalence of the two delegation-related classes with Wilson
           intervals; their overlap; prevalence by transport and by package
           registry (chi-square with Cramer's V, BH-corrected); comparison
           of remote-hosted servers against locally installed ones, since a
           remote server holds credentials on someone else's infrastructure.
 Output  : analysis/rq5_ambient_authority/outputs/rq5_results.{csv,json}
           rq5_apa_report.txt + figures/
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

RQ = "rq5"
OUT, FIG = config.rq_output_dir(RQ), config.rq_figure_dir(RQ)
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300})
BLUE, RED, GREY = "#2E5E8C", "#B0413E", "#7A7A7A"

FOCUS = ["AA_ambient_authority", "TPT_token_passthrough"]


def main():
    print("=" * 76)
    print(" RQ5 — AMBIENT AUTHORITY AND TOKEN PASSTHROUGH")
    print("=" * 76)
    df = load_master(config.FILE_MASTER)
    d = df[df["has_code"].astype(bool)].copy()
    n = len(d)
    print(f"  scan population : {n:,}")

    rb = ReportBuilder(RQ)
    pop = {"servers_scanned": n}
    if "has_remote" in d:
        pop["servers_with_remote_endpoint"] = int(d["has_remote"].fillna(0).sum())
    if "n_packages" in d:
        pop["servers_with_package"] = int((d["n_packages"].fillna(0) > 0).sum())
    rb.population(**pop)
    rb.text("Ambient authority and token passthrough both concern how a",
            "server handles authority it did not create: the first grants",
            "itself broad access, the second forwards a caller's credential",
            "onward. They are analysed together because both break the",
            "expectation that a tool acts only within a scope the user set.")

    # ---- prevalence + overlap ----------------------------------------------
    rows, present = [], []
    for cls in FOCUS:
        col = f"has_{cls}"
        if col not in d:
            continue
        k = int(d[col].fillna(0).sum())
        rows.append(st.prevalence(k, n, cls))
        present.append(cls)
        print(f"    {cls:<30s} {k:>6,}  {k/n:>7.2%}")
    cols = [f"has_{c}" for c in present]
    if len(cols) == 2:
        both = int(((d[cols[0]].fillna(0) > 0) &
                    (d[cols[1]].fillna(0) > 0)).sum())
        either = int(((d[cols[0]].fillna(0) > 0) |
                      (d[cols[1]].fillna(0) > 0)).sum())
        rows.append(st.prevalence(both, n, "both classes"))
        rows.append(st.prevalence(either, n, "either class"))
        print(f"    {'both':<30s} {both:>6,}   {'either':<8s} {either:>6,}")
    rb.prevalence_table(rows, "Delegation-risk prevalence (scan population)")

    # ---- by transport and registry -------------------------------------------
    tests = []
    for stratum, label in [("primary_transport", "transport"),
                           ("primary_package", "package registry")]:
        sub = d[d[stratum].notna()].copy()
        keep = sub[stratum].value_counts()
        keep = keep[keep >= 200].index.tolist()
        sub = sub[sub[stratum].isin(keep)]
        if sub[stratum].nunique() < 2:
            continue
        sub["delegation_risk"] = (
            sub[cols].fillna(0).sum(axis=1) > 0).astype(int)
        tab = pd.crosstab(sub[stratum], sub["delegation_risk"])
        if tab.shape[1] < 2:
            continue
        res = st.chi_square(tab.values)
        res["stratum"] = stratum
        tests.append(res)
        br = (sub.groupby(stratum)["delegation_risk"]
              .agg(["sum", "count"]).reset_index())
        br.columns = [stratum, "k", "n"]
        br["prevalence"] = br["k"] / br["n"]
        rb.dataframe_table(br.round(4), f"Delegation risk by {label}")
        print(f"    by {label:<18s} chi2={res['chi2']:.1f} "
              f"{st.format_p(res['p'])} V={res['cramers_v']:.3f}")

    # ---- remote vs local -------------------------------------------------------
    d["delegation_risk"] = (d[cols].fillna(0).sum(axis=1) > 0).astype(int)
    rem = d[d["has_remote"] == 1]
    loc = d[d["has_remote"] == 0]
    if len(rem) > 30 and len(loc) > 30:
        rl = st.two_proportion_test(
            int(rem["delegation_risk"].sum()), len(rem),
            int(loc["delegation_risk"].sum()), len(loc),
            "remote-hosted", "locally installed")
        tests.append(rl)
        rb.chi_square_result(rl, "Delegation risk, remote-hosted vs local")
        rb.text("", "A remote-hosted server holds any forwarded credential on",
                "infrastructure the user does not control, so the same",
                "pattern carries more consequence there than in a server the",
                "user runs locally.")
        print(f"    remote vs local OR="
              f"{rl.get('odds_ratio', float('nan')):.2f} "
              f"{st.format_p(rl['p'])}")

    if tests:
        p_adj = st.adjust_pvalues([t["p"] for t in tests])
        for t_, pa in zip(tests, p_adj):
            t_["p_adj"] = pa

    # ---- what the evidence looks like -------------------------------------------
    if config.FILE_FINDINGS.exists():
        fd = pd.read_csv(config.FILE_FINDINGS, low_memory=False)
        rb.section("Representative evidence")
        rng = np.random.default_rng(config.RANDOM_SEED)
        for cls in present:
            sub = fd[fd["threat_class"] == cls]
            if not len(sub):
                continue
            take = sub.iloc[rng.choice(len(sub), size=min(3, len(sub)),
                                       replace=False)]
            rb.text(f"{cls}:")
            for _, r in take.iterrows():
                rb.text(f"   {str(r.get('evidence',''))[:88]}")
        rb.text("", "Server identities are withheld; findings are reported in",
                "aggregate.")

    rb.methods_note()
    rb.detection_caveat()

    # ---- figures ------------------------------------------------------------------
    plot = pd.DataFrame([r for r in rows if r["k"] > 0])
    fig, ax = plt.subplots(figsize=(3.45, 1.8))
    for i, (_, r) in enumerate(plot.iterrows()):
        ax.plot([r["ci_low"] * 100, r["ci_high"] * 100], [i, i],
                color=BLUE, lw=1.3)
        ax.plot(r["prevalence"] * 100, i, "o", color=BLUE, ms=4.2,
                mec="white", mew=.5, zorder=5)
        ax.annotate(f"{r['prevalence']*100:.2f}%",
                    (r["ci_high"] * 100, i), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=6.4)
    ax.set_yticks(range(len(plot)))
    ax.set_yticklabels([g.replace("_", " ") for g in plot["group"]])
    ax.set_xlabel("servers affected (%), 95% CI")
    ax.grid(axis="x", color="#DDDDDD", lw=.45); ax.set_axisbelow(True)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "rq5_prevalence.pdf"); plt.close(fig)
    print(f"  [SAVED] figures -> {FIG}")

    save_csv(pd.DataFrame(rows), OUT / "rq5_results.csv", "rq5 prevalence")
    save_json({"n_scanned": n, "prevalence": rows, "tests": tests},
              OUT / "rq5_results.json", "rq5 results")
    save_txt(rb.render(), OUT / "rq5_apa_report.txt", "rq5 APA report")
    print("  RQ5 COMPLETE")


if __name__ == "__main__":
    main()
