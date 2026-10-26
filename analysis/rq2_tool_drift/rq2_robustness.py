"""
============================================================================
 rq2_robustness.py, RQ2: Robustness of the Drift and Security Association
============================================================================
 Question: Does the association between silent drift and high-severity
           findings survive confounders, shared code, and uneven publishers?
 Approach: Over multi-version servers with fetched code: the bivariate odds
           ratio; logistic models adjusting for log stars, log files, log
           repository age, and log version count with standard errors
           clustered by repository; the same model for any finding; one
           server per repository; standard errors clustered by GitHub owner;
           one server per owner (seeded draw plus repeated draws); and the
           concentration of drifting, high-severity servers among owners.
           For endpoint redirection: distinct registry namespaces, the share
           held by the largest namespace, and one server per namespace.
 Also     : RQ1 high-severity prevalence with one server per GitHub owner.
 Output  : analysis/rq2_tool_drift/outputs/rq2_robustness.{json,txt}
============================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from setup import config                                              # noqa: E402
from utils.io_helpers import load_master, save_json, save_txt         # noqa: E402
from utils import stats as st                                         # noqa: E402

OUT = config.rq_output_dir("rq2")
OUT.mkdir(parents=True, exist_ok=True)
FORMULA = "y ~ silent + log_stars + log_files + log_age + log_versions"


def odds_ratio_2x2(d: pd.DataFrame) -> dict:
    tab = pd.crosstab(d["silent"], d["y"]).reindex(index=[0, 1], columns=[0, 1],
                                                   fill_value=0)
    return st.chi_square(tab.values[::-1, ::-1])   # rows: drift yes/no; cols: y yes/no


def fit(d: pd.DataFrame, cluster: str = None) -> dict:
    """Logistic model; odds ratio for silent drift and for version count."""
    if cluster:
        groups = pd.factorize(d[cluster])[0]
        m = smf.logit(FORMULA, data=d).fit(disp=0, cov_type="cluster",
                                           cov_kwds={"groups": groups})
    else:
        m = smf.logit(FORMULA, data=d).fit(disp=0)
    ci = np.exp(m.conf_int())
    return {"n": int(m.nobs), "or": float(np.exp(m.params["silent"])),
            "ci": [float(ci.loc["silent", 0]), float(ci.loc["silent", 1])],
            "p": float(m.pvalues["silent"]), "pseudo_r2": float(m.prsquared),
            "or_versions": float(np.exp(m.params["log_versions"])),
            "p_versions": float(m.pvalues["log_versions"]),
            "cluster": cluster or "none"}


def fmt(r: dict) -> str:
    return f"OR = {r['or']:.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}], n = {r['n']:,}"


def main():
    print("=" * 76)
    print(" RQ2 — ROBUSTNESS OF THE DRIFT AND SECURITY ASSOCIATION")
    print("=" * 76)
    df = load_master(config.FILE_MASTER)
    d = df[df["has_code"].astype(bool) & (df["n_versions"].fillna(1) > 1)].copy()
    d["silent"] = d["has_silent_drift"].fillna(0).astype(int)
    d["y"] = d["has_high_severity"].fillna(0).astype(int)
    d["any_finding"] = (d["n_findings_total"].fillna(0) > 0).astype(int)
    d["log_stars"] = np.log1p(d["stars"].fillna(0))
    d["log_files"] = np.log1p(d["n_files"].fillna(0))
    d["log_age"] = np.log1p(d["repo_age_days"].fillna(0))
    d["log_versions"] = np.log1p(d["n_versions"])
    d["repo_key"] = d["owner"].astype(str) + "/" + d["repo"].astype(str)
    print(f"  multi-version servers with code : {len(d):,}  "
          f"repositories {d['repo_key'].nunique():,}  owners {d['owner'].nunique():,}")

    res = {"n": len(d), "repositories": int(d["repo_key"].nunique()),
           "owners": int(d["owner"].nunique())}
    lines = ["RQ2 robustness: silent drift and high-severity findings",
             f"Population: {len(d):,} multi-version servers with fetched code "
             f"({res['repositories']:,} repositories, {res['owners']:,} GitHub owners).", ""]

    rates = d.groupby("silent")["y"].mean()
    res["rate_silent"], res["rate_not_silent"] = float(rates[1]), float(rates[0])
    biv = odds_ratio_2x2(d)
    res["bivariate"] = biv
    lines.append(f"High-severity rate: {rates[1]:.1%} with silent drift, {rates[0]:.1%} without.")
    lines.append(f"Bivariate: OR = {biv['odds_ratio']:.2f} [{biv['or_ci_low']:.2f}, "
                 f"{biv['or_ci_high']:.2f}], chi2(1) = {biv['chi2']:.2f}, V = {biv['cramers_v']:.2f}")

    adj = fit(d, "repo_key")
    res["adjusted_repo_cluster"] = adj
    lines.append(f"Adjusted, clustered by repository: {fmt(adj)}, pseudo-R2 = {adj['pseudo_r2']:.3f}; "
                 f"version count OR = {adj['or_versions']:.2f}, p = {adj['p_versions']:.3f}")

    any_d = d.assign(y=d["any_finding"])
    adj_any = fit(any_d, "repo_key")
    res["adjusted_any_finding"] = adj_any
    lines.append(f"Adjusted, any finding as outcome: {fmt(adj_any)}")

    one_repo = d.drop_duplicates("repo_key")
    r_repo = fit(one_repo)
    res["one_per_repository"] = r_repo
    lines.append(f"One server per repository (first listed): {fmt(r_repo)}")

    adj_owner = fit(d, "owner")
    res["adjusted_owner_cluster"] = adj_owner
    lines.append(f"Adjusted, clustered by GitHub owner: {fmt(adj_owner)}")

    one_owner = d.sample(frac=1, random_state=config.RANDOM_SEED).drop_duplicates("owner")
    r_owner = fit(one_owner)
    res["one_per_owner"] = r_owner
    draws = [fit(d.sample(frac=1, random_state=i).drop_duplicates("owner"))["or"]
             for i in range(config.N_SUBSAMPLE_DRAWS)]
    res["one_per_owner_draws"] = {"draws": config.N_SUBSAMPLE_DRAWS,
                                  "median": float(np.median(draws)),
                                  "p2_5": float(np.percentile(draws, 2.5)),
                                  "p97_5": float(np.percentile(draws, 97.5))}
    lines.append(f"One server per GitHub owner (seed {config.RANDOM_SEED}): {fmt(r_owner)}; "
                 f"median over {config.N_SUBSAMPLE_DRAWS} draws {np.median(draws):.2f} "
                 f"[{np.percentile(draws, 2.5):.2f}, {np.percentile(draws, 97.5):.2f}]")

    both = d[(d["silent"] == 1) & (d["y"] == 1)]
    top2 = both["owner"].value_counts().head(2).index.tolist()
    t2 = d[d["owner"].isin(top2)]
    share = float(((t2["silent"] == 1) & (t2["y"] == 1)).mean())
    rest = fit(d[~d["owner"].isin(top2)], "repo_key")
    res["top_two_owners"] = {"servers": int(len(t2)),
                             "share_drift_and_high_severity": share,
                             "share_of_all_drift_and_high_severity":
                                 float(len(both[both["owner"].isin(top2)]) / len(both)),
                             "adjusted_without_them": rest}
    lines.append(f"The two owners with the most drifting, high-severity servers hold "
                 f"{len(t2):,} servers; {share:.0%} of those both drift and carry a "
                 f"high-severity finding. Without them: {fmt(rest)}")

    # ---- RQ1 prevalence with one server per owner -----------------------------
    scan = df[df["has_code"].astype(bool)].copy()
    scan["hs"] = scan["has_high_severity"].fillna(0)
    per_owner = [scan.sample(frac=1, random_state=i).drop_duplicates("owner")["hs"].mean()
                 for i in range(config.N_SUBSAMPLE_DRAWS)]
    res["rq1_high_severity_one_per_owner"] = {
        "per_server": float(scan["hs"].mean()), "owners": int(scan["owner"].nunique()),
        "median": float(np.median(per_owner)),
        "p2_5": float(np.percentile(per_owner, 2.5)),
        "p97_5": float(np.percentile(per_owner, 97.5))}
    lines += ["", f"RQ1 check: high-severity prevalence is {scan['hs'].mean():.1%} per server and "
              f"{np.median(per_owner):.1%} with one server per GitHub owner "
              f"({scan['owner'].nunique():,} owners; median over {config.N_SUBSAMPLE_DRAWS} draws)."]

    # ---- endpoint redirection by namespace ----------------------------------
    if config.FILE_DRIFT.exists():
        ev = pd.read_csv(config.FILE_DRIFT, low_memory=False)
        multi = df[df["n_versions"].fillna(1) > 1].copy()
        multi["ns"] = multi["name"].str.split("/").str[0]
        red = set(ev.loc[ev["change"] == "remote_host_redirected", "server_name"])
        multi["red"] = multi["name"].isin(red).astype(int)
        by_ns = multi[multi["red"] == 1]["ns"].value_counts()
        top_ns = by_ns.index[0]
        rest_ns = multi[multi["ns"] != top_ns]
        per_ns = [multi.sample(frac=1, random_state=i).drop_duplicates("ns")["red"].mean()
                  for i in range(config.N_SUBSAMPLE_DRAWS)]
        res["redirection"] = {"servers": int(multi["red"].sum()),
                              "namespaces": int(by_ns.size),
                              "largest_namespace_servers": int(by_ns.iloc[0]),
                              "rate_without_largest": float(rest_ns["red"].mean()),
                              "one_per_namespace_median": float(np.median(per_ns))}
        lines += ["", f"Redirection: {int(multi['red'].sum()):,} servers from {by_ns.size:,} "
                  f"registry namespaces; the largest holds {by_ns.iloc[0]:,}. Without it, "
                  f"{rest_ns['red'].mean():.2%} of multi-version servers; one server per "
                  f"namespace, median {np.median(per_ns):.2%} over {config.N_SUBSAMPLE_DRAWS} draws."]

    lines += ["", "All models are correlational; adjustment rules out the measured "
              "confounders only."]
    for ln in lines:
        print("    " + ln if ln else "")
    save_json(res, OUT / "rq2_robustness.json", "rq2 robustness")
    save_txt("\n".join(lines) + "\n", OUT / "rq2_robustness.txt", "rq2 robustness report")
    print("  RQ2 ROBUSTNESS COMPLETE")


if __name__ == "__main__":
    main()
