"""
============================================================================
 01_clean_and_merge.py, Normalize, Dedupe, Join Registry + GitHub Signals
============================================================================
 Project : Same Name, Different Server (MCP Security Census, IEEE BigData 2026)
 Purpose : Produce ONE clean server-level table: the latest version of each
           registry entry, joined to repository popularity and maintenance
           signals and to code-fetch status. This is the frame every later
           stage indexes against.
 Input   : data/raw/registry_servers.parquet, data/interim/code_index.csv
 Output  : data/interim/servers_merged.csv (+ .json summary + report txt)
 Usage   : python data_processing/01_clean_and_merge.py
============================================================================
"""

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setup import config                                    # noqa: E402
from utils.io_helpers import save_csv, save_json, save_txt  # noqa: E402


def parse_repo(url: str):
    m = re.match(r"https?://github\.com/([^/]+)/([^/\s#?]+)", str(url).strip())
    return (m.group(1), m.group(2).replace(".git", "")) if m else (None, None)


def main():
    print("=" * 76)
    print(" STEP 1/4 (PROCESSING) — CLEAN & MERGE")
    print("=" * 76)
    if not config.FILE_REGISTRY_RAW.exists():
        sys.exit("[FAIL] registry_servers.parquet missing — run collection 01")
    reg = pd.read_parquet(config.FILE_REGISTRY_RAW)
    print(f"  registry version records : {len(reg):,}")

    # ---- 1. one row per server (latest version) ---------------------------
    reg["published_dt"] = pd.to_datetime(reg["published_at"], utc=True,
                                         errors="coerce", format="mixed")
    latest = reg[reg["is_latest"]] if reg["is_latest"].any() else reg
    latest = (latest.sort_values("published_dt")
                    .drop_duplicates(subset="name", keep="last").copy())
    print(f"  unique servers (latest)  : {len(latest):,}")

    # ---- 2. version counts from the history table -------------------------
    if config.FILE_VERSIONS.exists():
        ver = pd.read_parquet(config.FILE_VERSIONS)
        vc = ver.groupby("name").agg(
            n_versions=("version", "count"),
            first_published=("published_at", "min"),
            last_published=("published_at", "max")).reset_index()
        latest = latest.merge(vc, on="name", how="left")
        print(f"  version history joined   : "
              f"{latest['n_versions'].notna().mean():.1%} of servers")
    else:
        latest["n_versions"] = np.nan
        print("  [WARN] server_versions.parquet missing — n_versions blank")

    # ---- 3. repository identity + code-fetch status -----------------------
    latest[["owner", "repo"]] = latest["repo_url"].apply(
        lambda u: pd.Series(parse_repo(u)))
    latest["has_repo"] = latest["owner"].notna()

    if config.FILE_CODE_INDEX.exists():
        idx = pd.read_csv(config.FILE_CODE_INDEX, low_memory=False)
        # repo-level rollup: status, file count, and GitHub signals
        repo_status = (idx.sort_values("status")
                          .drop_duplicates(["owner", "repo"], keep="first")
                          [["owner", "repo", "status", "stars", "forks_count"]
                           if "forks_count" in idx.columns else
                           ["owner", "repo", "status", "stars"]])
        files = (idx[idx["status"] == "ok"]
                 .groupby(["owner", "repo"])
                 .agg(n_files=("path", "count"),
                      bytes_fetched=("bytes", "sum")).reset_index())
        meta_cols = [c for c in ["stars", "pushed_at", "created_at", "language",
                                 "archived", "open_issues", "license"]
                     if c in idx.columns]
        repo_meta = (idx.drop_duplicates(["owner", "repo"])
                        [["owner", "repo"] + meta_cols])
        latest = (latest
                  .merge(repo_status[["owner", "repo", "status"]]
                         .rename(columns={"status": "code_status"}),
                         on=["owner", "repo"], how="left")
                  .merge(files, on=["owner", "repo"], how="left")
                  .merge(repo_meta, on=["owner", "repo"], how="left"))
        latest["has_code"] = latest["n_files"].fillna(0) > 0
        print(f"  code index joined        : "
              f"{latest['has_code'].mean():.1%} of servers have fetched code")
    else:
        latest["has_code"] = False
        print("  [WARN] code_index.csv missing — no code signals")

    # ---- 4. derived fields ------------------------------------------------
    now = pd.Timestamp.now(tz="UTC")
    for col, out in [("pushed_at", "days_since_push"),
                     ("created_at", "repo_age_days")]:
        if col in latest.columns:
            dt = pd.to_datetime(latest[col], utc=True, errors="coerce",
                                format="mixed")
            latest[out] = (now - dt).dt.days
    latest["stars"] = pd.to_numeric(latest.get("stars"), errors="coerce")
    latest["log_stars"] = np.log1p(latest["stars"])
    latest["is_deprecated"] = (latest["status"] == "deprecated").astype(int)
    latest["primary_package"] = (latest["package_types"].fillna("")
                                 .str.split(";").str[0].replace("", "none"))
    latest["primary_transport"] = (latest["remote_types"].fillna("")
                                   .str.split(";").str[0].replace("", "none"))
    latest["has_remote"] = (latest["n_remotes"].fillna(0) > 0).astype(int)

    keep = ["name", "version", "title", "description", "repo_url", "owner",
            "repo", "has_repo", "status", "is_deprecated", "published_at",
            "updated_at", "n_versions", "first_published", "last_published",
            "n_packages", "package_types", "primary_package", "n_remotes",
            "remote_types", "primary_transport", "has_remote", "code_status",
            "has_code", "n_files", "bytes_fetched", "stars", "log_stars",
            "language", "archived", "open_issues", "license",
            "days_since_push", "repo_age_days"]
    out = latest[[c for c in keep if c in latest.columns]].copy()
    save_csv(out, config.FILE_SERVERS_MERGED, "merged server table")

    # ---- report -----------------------------------------------------------
    cs = out["code_status"].value_counts(dropna=False).to_dict() \
        if "code_status" in out else {}
    lines = [
        "=" * 76, " CLEAN & MERGE REPORT",
        f" Generated : {datetime.now(timezone.utc).isoformat()}",
        f" Snapshot  : {config.SNAPSHOT_LABEL}", "=" * 76,
        f" Servers (latest version)   : {len(out):,}",
        f" With repository URL        : {int(out['has_repo'].sum()):,} "
        f"({out['has_repo'].mean():.1%})",
        f" With fetched code          : {int(out['has_code'].sum()):,} "
        f"({out['has_code'].mean():.1%})   <- scan population",
        f" Deprecated                 : {int(out['is_deprecated'].sum()):,}",
        f" Multi-version servers      : "
        f"{int((out['n_versions'].fillna(1) > 1).sum()):,}   <- RQ2 set",
        "-" * 76,
        f" Stars   median / mean / max: {out['stars'].median():.0f} / "
        f"{out['stars'].mean():.0f} / {out['stars'].max():.0f}",
        f" Zero-star servers          : {(out['stars'] == 0).sum():,} "
        f"({(out['stars'] == 0).mean():.1%})",
        f" Archived repositories      : {int(out['archived'].fillna(0).sum()):,}",
        f" Median days since push     : {out['days_since_push'].median():.0f}",
        "-" * 76, " Code fetch outcome:",
    ]
    for k, v in cs.items():
        lines.append(f"   {str(k):<24s} {v:>8,}")
    lines.append(" Primary package type:")
    for k, v in out["primary_package"].value_counts().head(8).items():
        lines.append(f"   {k:<24s} {v:>8,}")
    lines.append(" Primary transport:")
    for k, v in out["primary_transport"].value_counts().head(6).items():
        lines.append(f"   {k:<24s} {v:>8,}")
    lines.append("=" * 76)

    save_json({"n_servers": len(out),
               "with_code": int(out["has_code"].sum()),
               "code_status_counts": {str(k): int(v) for k, v in cs.items()},
               "generated_at_utc": datetime.now(timezone.utc).isoformat()},
              config.DATA_INTERIM / "servers_merged_summary.json", "summary")
    save_txt("\n".join(lines), config.DATA_INTERIM / "merge_report.txt",
             "merge report")
    print("\n".join(lines))
    print("  Next: python data_processing/02_scan_security.py")


if __name__ == "__main__":
    main()
