"""
============================================================================
 04_feature_engineering.py, Build the Master Analysis Table
============================================================================
 Project : Same Name, Different Server (MCP Security Census, IEEE BigData 2026)
 Purpose : Join server metadata, scan findings, and drift events into ONE
           server-level table with every feature the five RQs need:
             RQ1  per-class flags and counts, severity score
             RQ2  drift counts and silent-drift flag
             RQ3  popularity and maintenance signals
             RQ4  scanner outputs to compare against manual labels
             RQ5  ambient-authority and token-passthrough indicators
 Input   : servers_merged.csv, security_findings.csv, drift_events.csv,
           tool_definitions.csv
 Output  : data/processed/master.csv (+ master_summary.json + report txt)
 Usage   : python data_processing/04_feature_engineering.py
============================================================================
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setup import config                                    # noqa: E402
from utils.io_helpers import save_csv, save_json, save_txt  # noqa: E402


def main():
    print("=" * 76)
    print(" STEP 4/4 (PROCESSING) — FEATURE ENGINEERING -> MASTER TABLE")
    print("=" * 76)
    if not config.FILE_SERVERS_MERGED.exists():
        sys.exit("[FAIL] servers_merged.csv missing — run processing 01")
    df = pd.read_csv(config.FILE_SERVERS_MERGED, low_memory=False)
    print(f"  servers               : {len(df):,}")

    # ---- findings ---------------------------------------------------------
    if config.FILE_FINDINGS.exists():
        fd = pd.read_csv(config.FILE_FINDINGS, low_memory=False)
        if len(fd):
            # per-class counts, wide
            counts = (fd.groupby(["server_name", "threat_class"]).size()
                        .unstack(fill_value=0))
            counts.columns = [f"n_{c}" for c in counts.columns]
            df = df.merge(counts.reset_index(),
                          left_on="name", right_on="server_name", how="left")
            df.drop(columns=["server_name"], errors="ignore", inplace=True)
            # ensure every catalogue class has a column even if never hit
            for cls in config.THREAT_CLASSES:
                col = f"n_{cls}"
                if col not in df.columns:
                    df[col] = 0
            n_cols = [f"n_{c}" for c in config.THREAT_CLASSES]
            df[n_cols] = df[n_cols].fillna(0)
            # only servers we actually scanned can carry a zero; others NaN
            df.loc[~df["has_code"].astype(bool), n_cols] = np.nan
            for cls in config.THREAT_CLASSES:
                df[f"has_{cls}"] = (df[f"n_{cls}"] > 0).astype("Int64")
            df["n_findings_total"] = df[n_cols].sum(axis=1, min_count=1)
            df["n_classes_hit"] = (df[n_cols] > 0).sum(axis=1)
            df.loc[df[n_cols].isna().all(axis=1), "n_classes_hit"] = np.nan
            # weighted severity score
            df["severity_score"] = sum(
                df[f"has_{cls}"].fillna(0).astype(float)
                * config.SEVERITY_WEIGHT[spec["severity"]]
                for cls, spec in config.THREAT_CLASSES.items())
            df.loc[~df["has_code"].astype(bool), "severity_score"] = np.nan
            high = [f"has_{c}" for c, s in config.THREAT_CLASSES.items()
                    if s["severity"] == "high"]
            df["has_high_severity"] = (df[high].fillna(0).sum(axis=1) > 0
                                       ).astype("Int64")
            df.loc[~df["has_code"].astype(bool), "has_high_severity"] = pd.NA
            df["any_finding"] = (df["n_findings_total"] > 0).astype("Int64")
            print(f"  findings joined       : "
                  f"{int(df['any_finding'].fillna(0).sum()):,} servers flagged")
    else:
        print("  [WARN] security_findings.csv missing")

    # ---- tool definitions -------------------------------------------------
    tool_file = config.DATA_INTERIM / "tool_definitions.csv"
    if tool_file.exists():
        td = pd.read_csv(tool_file, low_memory=False)
        if len(td):
            agg = (td.groupby("server_name")
                     .agg(n_tools_found=("tool_name", "nunique"),
                          mean_tool_desc_len=("tool_description",
                                              lambda s: s.astype(str)
                                              .str.len().mean()))
                     .reset_index())
            df = df.merge(agg, left_on="name", right_on="server_name",
                          how="left").drop(columns=["server_name"],
                                           errors="ignore")
            df["n_tools_found"] = df["n_tools_found"].fillna(0)
            print(f"  tool definitions      : "
                  f"{int((df['n_tools_found'] > 0).sum()):,} servers with tools")

    # ---- drift ------------------------------------------------------------
    if config.FILE_DRIFT.exists():
        dr = pd.read_csv(config.FILE_DRIFT, low_memory=False)
        if len(dr):
            agg = dr.groupby("server_name").agg(
                n_drift_events=("change", "count"),
                n_silent_drift=("drift_type",
                                lambda s: int((s == "silent_semantic").sum())),
                n_breaking_drift=("drift_type",
                                  lambda s: int((s == "breaking").sum())),
                median_days_between=("days_between", "median")).reset_index()
            df = df.merge(agg, left_on="name", right_on="server_name",
                          how="left").drop(columns=["server_name"],
                                           errors="ignore")
            multi = df["n_versions"].fillna(1) > 1
            for c in ["n_drift_events", "n_silent_drift", "n_breaking_drift"]:
                df.loc[multi, c] = df.loc[multi, c].fillna(0)
            df["has_silent_drift"] = np.where(
                multi, (df["n_silent_drift"].fillna(0) > 0).astype(float),
                np.nan)
            print(f"  drift joined          : "
                  f"{int(df['has_silent_drift'].fillna(0).sum()):,} servers "
                  f"with silent drift")

    # ---- derived analysis fields -----------------------------------------
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df["popularity_decile"] = pd.qcut(df["stars"].rank(method="first"),
                                      10, labels=False, duplicates="drop") + 1
    df["is_popular"] = (df["stars"] >= 10).astype("Int64")
    df["is_maintained"] = (df["days_since_push"] <= 180).astype("Int64")
    df["is_archived"] = df["archived"].fillna(False).astype(int)
    df["has_license"] = df["license"].notna().astype(int)
    df["findings_per_file"] = np.where(
        df["n_files"].fillna(0) > 0,
        df["n_findings_total"] / df["n_files"], np.nan)

    save_csv(df, config.FILE_MASTER, "MASTER table")

    # ---- report -----------------------------------------------------------
    scanned = df[df["has_code"].astype(bool)]
    lines = [
        "=" * 76, " MASTER TABLE REPORT",
        f" Generated : {datetime.now(timezone.utc).isoformat()}",
        f" Snapshot  : {config.SNAPSHOT_LABEL}", "=" * 76,
        f" Rows (servers)          : {len(df):,}",
        f" Features                : {len(df.columns)}",
        f" Scanned (has code)      : {len(scanned):,}",
        "-" * 76,
        f" {'Threat class':<32s} {'servers':>9s} {'prevalence':>12s}",
    ]
    for cls in config.THREAT_CLASSES:
        col = f"has_{cls}"
        if col in df.columns and len(scanned):
            k = int(scanned[col].fillna(0).sum())
            lines.append(f" {cls:<32s} {k:>9,} "
                         f"{k/len(scanned):>11.2%}")
    if "has_high_severity" in df.columns and len(scanned):
        k = int(scanned["has_high_severity"].fillna(0).sum())
        lines += ["-" * 76,
                  f" Any high-severity class : {k:,} ({k/len(scanned):.2%})"]
    if "has_silent_drift" in df.columns:
        m = df["has_silent_drift"].notna()
        if m.any():
            k = int(df.loc[m, "has_silent_drift"].sum())
            lines.append(f" Silent drift (multi-ver): {k:,} "
                         f"({k/m.sum():.2%} of {m.sum():,})")
    lines += ["=" * 76]

    save_json({"n_servers": int(len(df)), "n_features": int(len(df.columns)),
               "n_scanned": int(len(scanned)),
               "feature_columns": list(df.columns),
               "generated_at_utc": datetime.now(timezone.utc).isoformat()},
              config.FILE_MASTER_JSON, "master summary")
    save_txt("\n".join(lines), config.DATA_PROCESSED / "master_report.txt",
             "master report")
    print("\n".join(lines))
    print("  Next: python data_validation/validate_dataset.py")


if __name__ == "__main__":
    main()
