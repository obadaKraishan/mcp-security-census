"""
============================================================================
 run_all.py, One-Command Pipeline Orchestrator
============================================================================
 Project : Same Name, Different Server (MCP Security Census, IEEE BigData 2026)
 Purpose : Run the study in order and stop on the first failure:
             collection -> processing -> validation -> codebook -> labeling
             scoring -> all five analyses.
 Usage   : python run_all.py                  # everything (code fetch ~8 h once)
           python run_all.py --analysis-only  # validation, codebook, scoring,
                                              # analyses on the shipped tables
           python run_all.py --skip-fetch     # everything except the long
                                              # GitHub code fetch
============================================================================
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

COLLECTION = [
    ["python", "data_collection/01_harvest_registry.py"],
    ["python", "data_collection/02_fetch_version_history.py"],
    ["python", "data_collection/03_fetch_server_code.py"],
]
PROCESSING = [
    ["python", "data_processing/01_clean_and_merge.py"],
    ["python", "data_processing/02_scan_security.py"],
    ["python", "data_processing/03_detect_drift.py"],
    ["python", "data_processing/04_feature_engineering.py"],
]
GATE = [
    ["python", "data_validation/validate_dataset.py"],
    ["python", "codebook/generate_codebook.py"],
    ["python", "labeling/02_score_scanner.py"],
]
ANALYSES = [
    ["python", "analysis/rq1_threat_prevalence/rq1_analysis.py"],
    ["python", "analysis/rq2_tool_drift/rq2_analysis.py"],
    ["python", "analysis/rq2_tool_drift/rq2_robustness.py"],
    ["python", "analysis/rq3_popularity_security/rq3_analysis.py"],
    ["python", "analysis/rq4_scanner_accuracy/rq4_analysis.py"],
    ["python", "analysis/rq5_ambient_authority/rq5_analysis.py"],
]


def run(cmd: list[str]) -> None:
    print("\n" + "=" * 76)
    print(f" RUN: {' '.join(cmd)}")
    print("=" * 76, flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=ROOT)
    dt = (time.time() - t0) / 60
    if rc != 0:
        print(f"\n[ABORT] step failed (exit {rc}) after {dt:.1f} min: "
              f"{' '.join(cmd)}")
        sys.exit(rc)
    print(f"[ OK ] finished in {dt:.1f} min")


def main():
    ap = argparse.ArgumentParser(description="Run the full census pipeline")
    ap.add_argument("--analysis-only", action="store_true",
                    help="only validation, codebook, scoring, and the analyses")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="run everything except the long GitHub code fetch")
    args = ap.parse_args()

    t0 = time.time()
    if not args.analysis_only:
        steps = COLLECTION + PROCESSING
        if args.skip_fetch:
            steps = [s for s in steps if "03_fetch_server_code" not in s[1]]
        for cmd in steps:
            run(cmd)
    for cmd in GATE + ANALYSES:
        run(cmd)
    print("\n" + "=" * 76)
    print(f" PIPELINE COMPLETE in {(time.time()-t0)/60:.1f} min")
    print(" Results: analysis/rq*/outputs/  |  figures: analysis/rq*/figures/")
    print("=" * 76)


if __name__ == "__main__":
    main()
