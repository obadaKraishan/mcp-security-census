"""
============================================================================
 validate_dataset.py, Quality Gate Before Any Analysis Runs
============================================================================
 Project : Same Name, Different Server (MCP Security Census, IEEE BigData 2026)
 Purpose : Assert the master table is analysis-ready. Beyond schema and
           range checks, this gate guards two things specific to this study:
             1. NaN discipline. Servers we never scanned must be NaN on
                finding columns, never zero, or every prevalence denominator
                silently inflates.
             2. Evidence review. For each threat class it prints real
                evidence lines so the analyst can eyeball whether a high
                prevalence is genuine before it becomes a headline number.
 Input   : data/processed/master.csv, data/interim/security_findings.csv,
           data/interim/drift_events.csv
 Output  : data/processed/validation_report.txt / .json (+ console)
 Usage   : python data_validation/validate_dataset.py
           python data_validation/validate_dataset.py --evidence 5
           python data_validation/validate_dataset.py --no-evidence
============================================================================
 Exits nonzero if any blocking check fails.
============================================================================
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setup import config                                       # noqa: E402
from utils.io_helpers import load_master, save_json, save_txt  # noqa: E402
from utils.stats import wilson_ci                              # noqa: E402

PASS, FAIL, WARN = "[ OK ]", "[FAIL]", "[WARN]"
lines, failures, warnings_ = [], [], []


def log(msg=""):
    print(msg)
    lines.append(msg)


def section(t):
    log()
    log("-" * 76)
    log(f" {t}")
    log("-" * 76)


def check(name, ok, detail, blocking=True):
    tag = PASS if ok else (FAIL if blocking else WARN)
    log(f"  {tag} {name:<44s} {detail}")
    if not ok:
        (failures if blocking else warnings_).append(f"{name}: {detail}")


def main():
    ap = argparse.ArgumentParser(description="Validate the master table")
    ap.add_argument("--evidence", type=int, default=3,
                    help="evidence lines to print per threat class (0 = none)")
    ap.add_argument("--no-evidence", action="store_true")
    args = ap.parse_args()
    n_evidence = 0 if args.no_evidence else args.evidence

    log("=" * 76)
    log(" VALIDATION GATE — master.csv")
    log(f" Snapshot: {config.SNAPSHOT_LABEL}   "
        f"Run: {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S UTC}")
    log("=" * 76)
    df = load_master(config.FILE_MASTER)

    # ---- 1. schema ---------------------------------------------------------
    section("1/6  Schema and identity")
    required = ["name", "has_repo", "has_code", "status", "stars",
                "n_versions", "primary_package", "primary_transport"]
    missing = [c for c in required if c not in df.columns]
    check("required columns present", not missing,
          f"missing: {missing}" if missing else f"{len(df.columns)} columns")
    dup = df["name"].duplicated().sum()
    check("server names unique", dup == 0, f"{dup} duplicates")
    check("row count plausible", len(df) > 1000, f"{len(df):,} servers")

    # ---- 2. NaN discipline (the subtle one) --------------------------------
    section("2/6  NaN discipline on finding columns")
    scanned = df["has_code"].astype(bool)
    check("scan population non-empty", scanned.sum() > 0,
          f"{int(scanned.sum()):,} servers have fetched code")
    bad_cols = []
    for cls in config.THREAT_CLASSES:
        col = f"n_{cls}"
        if col not in df.columns:
            bad_cols.append(f"{col} absent")
            continue
        leaked = df.loc[~scanned, col].notna().sum()
        if leaked:
            bad_cols.append(f"{col}: {leaked} unscanned rows carry a value")
    check("unscanned servers are NaN, not zero", not bad_cols,
          "; ".join(bad_cols) if bad_cols else
          "no zero-leakage into unscanned rows")
    if "has_high_severity" in df.columns:
        leak = df.loc[~scanned, "has_high_severity"].notna().sum()
        check("has_high_severity NaN when unscanned", leak == 0,
              f"{leak} leaked rows")
    if "has_silent_drift" in df.columns:
        single = df["n_versions"].fillna(1) <= 1
        leak = df.loc[single, "has_silent_drift"].notna().sum()
        check("drift NaN for single-version servers", leak == 0,
              f"{leak} leaked rows", blocking=False)

    # ---- 3. ranges ---------------------------------------------------------
    section("3/6  Value ranges")
    checks = [("stars", 0, None), ("n_versions", 1, None),
              ("n_files", 0, None), ("severity_score", 0, 40),
              ("days_since_push", 0, None), ("popularity_decile", 1, 10)]
    for col, lo, hi in checks:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        bad = ((s < lo).sum() if lo is not None else 0) + \
              ((s > hi).sum() if hi is not None else 0)
        check(f"{col} within [{lo}, {hi if hi is not None else 'inf'}]",
              bad == 0, f"{bad} out of range")
    for cls in config.THREAT_CLASSES:
        col = f"has_{cls}"
        if col in df.columns:
            vals = set(pd.to_numeric(df[col], errors="coerce")
                       .dropna().unique().tolist())
            if not vals <= {0, 1}:
                check(f"{col} is binary", False, f"values {sorted(vals)[:5]}")
    check("finding flags binary", True, "all has_* columns are 0/1 or NaN")

    # ---- 4. internal consistency ------------------------------------------
    section("4/6  Internal consistency")
    if config.FILE_FINDINGS.exists():
        fd = pd.read_csv(config.FILE_FINDINGS, low_memory=False)
        for cls in list(config.THREAT_CLASSES)[:3]:
            col = f"has_{cls}"
            if col not in df.columns:
                continue
            from_master = int(df.loc[scanned, col].fillna(0).sum())
            from_source = fd.loc[fd["threat_class"] == cls,
                                 "server_name"].nunique()
            check(f"{cls} count matches findings file",
                  abs(from_master - from_source) <= 1,
                  f"master {from_master:,} vs findings {from_source:,}")
    if "n_findings_total" in df.columns and "n_classes_hit" in df.columns:
        bad = (df["n_classes_hit"].fillna(0) >
               df["n_findings_total"].fillna(0)).sum()
        check("classes_hit <= findings_total", bad == 0, f"{bad} rows violate")

    # ---- 5. per-class prevalence with intervals ---------------------------
    section("5/6  Threat-class prevalence (scan population)")
    n = int(scanned.sum())
    log(f"  {'Class':<30s} {'k':>7s} {'prevalence':>12s} {'95% CI':>18s}")
    log("  " + "-" * 70)
    prev_rows = []
    for cls in config.THREAT_CLASSES:
        col = f"has_{cls}"
        if col not in df.columns:
            continue
        k = int(df.loc[scanned, col].fillna(0).sum())
        lo, hi = wilson_ci(k, n)
        log(f"  {cls:<30s} {k:>7,} {k/n:>11.2%} "
            f"     [{lo*100:5.2f}, {hi*100:5.2f}]")
        prev_rows.append({"class": cls, "k": k, "n": n, "prevalence": k / n})
        if k / n > 0.50:
            warnings_.append(f"{cls} exceeds 50% prevalence — inspect patterns")

    # ---- 6. evidence review ------------------------------------------------
    if n_evidence and config.FILE_FINDINGS.exists():
        section(f"6/6  Evidence review ({n_evidence} samples per class)")
        log("  Read these before trusting the prevalence above. If a line")
        log("  looks like ordinary configuration rather than a risk, the")
        log("  pattern needs tightening.")
        fd = pd.read_csv(config.FILE_FINDINGS, low_memory=False)
        rng = np.random.default_rng(config.RANDOM_SEED)
        for cls in config.THREAT_CLASSES:
            sub = fd[fd["threat_class"] == cls]
            log("")
            log(f"  {cls}  ({len(sub):,} findings, "
                f"{sub['server_name'].nunique():,} servers)")
            if not len(sub):
                log("      (no findings)")
                continue
            take = sub.iloc[rng.choice(len(sub),
                                       size=min(n_evidence, len(sub)),
                                       replace=False)]
            for _, r in take.iterrows():
                ev = str(r.get("evidence", ""))[:96]
                log(f"      {str(r.get('file',''))[-34:]:<34s} | {ev}")
    else:
        section("6/6  Evidence review")
        log("  (skipped)")

    # ---- verdict -----------------------------------------------------------
    section("SUMMARY")
    if failures:
        log(f"  {FAIL} {len(failures)} blocking issue(s):")
        for f in failures:
            log(f"     - {f}")
    else:
        log(f"  {PASS} No blocking issues.")
    if warnings_:
        log(f"  {WARN} {len(warnings_)} warning(s):")
        for w in warnings_:
            log(f"     - {w}")
    verdict = ("MASTER TABLE VALID — safe to run analyses" if not failures
               else "VALIDATION FAILED — fix before running analyses")
    log()
    log("=" * 76)
    log(f" {verdict}")
    log("=" * 76)

    save_txt("\n".join(lines), config.DATA_PROCESSED / "validation_report.txt",
             "validation report")
    save_json({"generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "n_servers": int(len(df)), "n_scanned": n,
               "failures": failures, "warnings": warnings_,
               "prevalence": prev_rows},
              config.DATA_PROCESSED / "validation_report.json", "validation")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
