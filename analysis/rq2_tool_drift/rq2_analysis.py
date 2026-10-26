"""
============================================================================
 rq2_analysis.py, RQ2: Silent Drift in What a Server Advertises
============================================================================
 Question: How often do server definitions change silently across published
           versions?
 Approach: Over multi-version servers, prevalence of each drift type with
           Wilson intervals; endpoint-host redirection isolated as the
           closest observable analogue of a rug pull; timing of change (days
           from first publication to the first change, and days between
           consecutive changed versions); association between drift and
           security posture (chi-square with odds ratios, BH-corrected);
           robustness of redirection under a per-server transition cap.
           Adjusted models and publisher-level checks: rq2_robustness.py.
 Output  : analysis/rq2_tool_drift/outputs/rq2_results.{csv,json}
           rq2_apa_report.txt + figures/
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

RQ = "rq2"
OUT, FIG = config.rq_output_dir(RQ), config.rq_figure_dir(RQ)
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300})
BLUE, RED, GREY = "#2E5E8C", "#B0413E", "#7A7A7A"


def main():
    print("=" * 76)
    print(" RQ2 — VERSION DRIFT")
    print("=" * 76)
    df = load_master(config.FILE_MASTER)
    if not config.FILE_DRIFT.exists():
        sys.exit("[FAIL] drift_events.csv missing — run processing 03")
    ev = pd.read_csv(config.FILE_DRIFT, low_memory=False)
    multi = df[df["n_versions"].fillna(1) > 1].copy()
    M = len(multi)
    print(f"  multi-version servers : {M:,}")
    print(f"  drift events          : {len(ev):,}")

    rb = ReportBuilder(RQ)
    rb.population(servers_in_registry=len(df),
                  multi_version_servers=M,
                  drift_events=len(ev),
                  servers_with_any_drift=int(ev["server_name"].nunique()))
    rb.text("Drift is measured on the fields the registry versions:",
            "description text, package identifiers, and remote endpoints.",
            "The registry does not version tool definitions, so a client",
            "cannot detect a change in tool behaviour from the registry",
            "alone; that gap is reported as a structural finding.")

    # ---- prevalence of each drift type ------------------------------------
    rows = []
    for dtype in ["silent_semantic", "breaking", "additive"]:
        k = ev.loc[ev["drift_type"] == dtype, "server_name"].nunique()
        rows.append(st.prevalence(int(k), M, dtype))
    k_any = ev["server_name"].nunique()
    rows.append(st.prevalence(int(k_any), M, "any drift"))
    rb.prevalence_table(rows, "Drift prevalence (multi-version servers)")
    for r in rows:
        print(f"    {r['group']:<20s} {r['k']:>6,}  {r['prevalence']:>7.2%}")

    # ---- change kinds, with the rug-pull signal isolated -------------------
    kinds = []
    for kind in sorted(ev["change"].unique()):
        k = ev.loc[ev["change"] == kind, "server_name"].nunique()
        p = st.prevalence(int(k), M, kind)
        p["events"] = int((ev["change"] == kind).sum())
        kinds.append(p)
    kdf = pd.DataFrame(kinds).sort_values("prevalence", ascending=False)
    rb.prevalence_table(kdf.to_dict("records"),
                        "Change kinds (share of multi-version servers)")

    redirect = ev[ev["change"] == "remote_host_redirected"]
    n_red = redirect["server_name"].nunique()
    lo, hi = st.wilson_ci(int(n_red), M)
    rb.section("Endpoint redirection (rug-pull analogue)")
    rb.text(f"Servers that kept their registry identity while moving their",
            f"remote endpoint to a different host: {n_red:,} of {M:,} "
            f"({n_red/M:.2%}, 95% CI [{lo*100:.2f}%, {hi*100:.2f}%]).",
            f"Events observed: {len(redirect):,}.",
            "An already-installed client continues to resolve the same server",
            "name while its traffic reaches a different origin, and the",
            "protocol provides no notification of the change.")
    print(f"    endpoint redirected  {n_red:>6,}  {n_red/M:>7.2%}")

    # ---- timing of change ----------------------------------------------------
    # Two clocks, reported separately: (a) days from a server's first
    # publication to its first change event; (b) days between consecutive
    # versions whose transition carried a change.
    timing = {}
    if "days_between" in ev.columns and ev["days_between"].notna().any():
        to_pub = pd.to_datetime(ev["to_published"], utc=True, errors="coerce")
        first_change = to_pub.groupby(ev["server_name"]).min()
        first_pub = pd.to_datetime(multi.set_index("name")["first_published"],
                                   utc=True, errors="coerce")
        to_first = ((first_change - first_pub.reindex(first_change.index))
                    .dt.total_seconds() / 86400).dropna()
        gaps = (ev.drop_duplicates(["server_name", "from_version", "to_version"])
                ["days_between"].dropna())
        d_first = st.describe(to_first, "to first change")
        d_gap = st.describe(gaps, "between changes")
        within_day = float((to_first <= 1).mean())
        rb.descriptives_table([d_first, d_gap], "Timing of change in days (servers that changed)")
        rb.text("'To first change' runs from a server's first publication to its",
                "first change event; 'between changes' is the gap between",
                "consecutive versions whose transition carried a change.",
                f"Changed within one day of first publication: {within_day:.1%} "
                f"of {len(to_first):,} servers.")
        timing = {"to_first_change": d_first, "between_changed_versions": d_gap,
                  "share_within_one_day": within_day}
        print(f"    days to first change  median {d_first['median']:.1f} "
              f"(mean {d_first['mean']:.1f}); within a day {within_day:.1%}")
        print(f"    days between changes  median {d_gap['median']:.1f}")

    # ---- drift vs security posture ------------------------------------------
    tests = []
    joined = multi.merge(
        ev.groupby("server_name")["drift_type"]
          .apply(lambda s: int((s == "silent_semantic").any()))
          .rename("silent").reset_index(),
        left_on="name", right_on="server_name", how="left")
    joined["silent"] = joined["silent"].fillna(0).astype(int)
    scanned = joined[joined["has_code"].astype(bool)]
    for outcome, label in [("has_high_severity", "any high-severity finding"),
                           ("any_finding", "any finding")]:
        if outcome not in scanned:
            continue
        sub = scanned[scanned[outcome].notna()]
        if sub["silent"].nunique() < 2 or sub[outcome].nunique() < 2:
            continue
        tab = pd.crosstab(sub["silent"], sub[outcome].astype(int))
        res = st.chi_square(tab.values)
        res["outcome"] = outcome
        tests.append(res)
        print(f"    silent drift vs {label:<26s} "
              f"OR={res.get('odds_ratio', float('nan')):.2f} "
              f"{st.format_p(res['p'])}")
    if tests:
        p_adj = st.adjust_pvalues([t["p"] for t in tests])
        for t_, pa in zip(tests, p_adj):
            rb.chi_square_result(
                t_, f"Silent drift vs {t_['outcome']}", p_adj=pa)

    # ---- robustness: bulk republishers ---------------------------------------
    # Recompute redirection keeping only each server's first N transitions,
    # ordered by publication time in the raw version history.
    cap = {}
    if config.FILE_VERSIONS.exists():
        v = pd.read_parquet(config.FILE_VERSIONS,
                            columns=["name", "version", "published_at"])
        v["t"] = pd.to_datetime(v["published_at"], utc=True, errors="coerce")
        v = v.sort_values(["name", "t"]).drop_duplicates(["name", "version"])
        v["idx"] = v.groupby("name").cumcount()     # transition k ends at idx k
        rd = redirect.merge(v[["name", "version", "idx"]], how="left",
                            left_on=["server_name", "to_version"],
                            right_on=["name", "version"])
        n_cap = int(rd.loc[rd["idx"] <= config.DRIFT_TRANSITION_CAP,
                           "server_name"].nunique())
        shift = (n_red - n_cap) / M * 100
        cap = {"cap": config.DRIFT_TRANSITION_CAP, "redirected_full": int(n_red),
               "redirected_capped": n_cap, "shift_points": shift}
        rb.section("Robustness to bulk republishers")
        rb.text(f"Keeping at most {config.DRIFT_TRANSITION_CAP} transitions per "
                f"server, {n_cap:,} servers show a redirection ({n_cap / M:.2%}) "
                f"against {n_red:,} ({n_red / M:.2%}) without the cap, a shift of "
                f"{shift:.2f} percentage points.")
        print(f"    redirect, cap {config.DRIFT_TRANSITION_CAP:<3d}    {n_cap:>6,}  "
              f"shift {shift:.2f} pts")
    rb.methods_note()

    # ---- figures --------------------------------------------------------------
    plot = kdf[kdf["k"] > 0].sort_values("prevalence").tail(9)
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    for i, (_, r) in enumerate(plot.iterrows()):
        c = RED if "redirect" in r["group"] or "removed" in r["group"] else BLUE
        ax.plot([r["ci_low"] * 100, r["ci_high"] * 100], [i, i], color=c, lw=1.3)
        ax.plot(r["prevalence"] * 100, i, "o", color=c, ms=4.2,
                mec="white", mew=.5, zorder=5)
    ax.set_yticks(range(len(plot)))
    ax.set_yticklabels([g.replace("_", " ") for g in plot["group"]])
    ax.set_xlabel("multi-version servers affected (%), 95% CI")
    ax.grid(axis="x", color="#DDDDDD", lw=.45); ax.set_axisbelow(True)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "rq2_drift_kinds.pdf"); plt.close(fig)

    surf = ev.groupby(["surface", "drift_type"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(3.45, 1.9))
    surf.plot(kind="bar", stacked=True, ax=ax,
              color=[GREY, RED, BLUE][:surf.shape[1]], width=.6)
    ax.set_xlabel(""); ax.set_ylabel("events")
    ax.legend(frameon=False, fontsize=6.2)
    plt.xticks(rotation=0)
    ax.grid(axis="y", color="#DDDDDD", lw=.45); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(FIG / "rq2_surface.pdf"); plt.close(fig)
    print(f"  [SAVED] figures -> {FIG}")

    save_csv(kdf, OUT / "rq2_results.csv", "rq2 drift kinds")
    save_json({"multi_version_servers": M,
               "drift_type_prevalence": rows,
               "change_kinds": kdf.to_dict("records"),
               "endpoint_redirect": {"k": int(n_red), "n": M,
                                     "ci": [lo, hi],
                                     "events": int(len(redirect))},
               "security_association": tests,
               "timing": timing,
               "transition_cap": cap},
              OUT / "rq2_results.json", "rq2 results")
    save_txt(rb.render(), OUT / "rq2_apa_report.txt", "rq2 APA report")
    print("  RQ2 COMPLETE")


if __name__ == "__main__":
    main()
