"""
============================================================================
 generate_codebook.py, Variable Dictionary for the Master Table
============================================================================
 Project : Same Name, Different Server (MCP Security Census, IEEE BigData 2026)
 Purpose : Emit a codebook documenting every master.csv
           variable: definition, type, unit, source stage, and coverage.
           Also emits the threat catalogue as its own table, since that is
           the Method section of the paper.
           Anything undocumented is flagged loudly so the codebook cannot
           silently drift from the data.
 Input   : data/processed/master.csv
 Output  : codebook/codebook.csv / .json / .txt
           codebook/threat_catalogue.csv / .txt
 Usage   : python codebook/generate_codebook.py
============================================================================
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setup import config                                              # noqa: E402
from utils.io_helpers import load_master, save_csv, save_json, save_txt  # noqa: E402

# (definition, unit, source stage)
D = {
    # identity and registry metadata
    "name": ("Canonical server name in the MCP registry", "-", "registry"),
    "version": ("Latest published version string", "-", "registry"),
    "title": ("Human-readable server title", "-", "registry"),
    "description": ("Server description text", "-", "registry"),
    "repo_url": ("Declared source repository URL", "-", "registry"),
    "owner": ("GitHub owner parsed from repo_url", "-", "clean_and_merge"),
    "repo": ("GitHub repository name parsed from repo_url", "-", "clean_and_merge"),
    "has_repo": ("1 if the record declares a repository URL", "0/1", "clean_and_merge"),
    "status": ("Registry status (active/deprecated)", "-", "registry"),
    "is_deprecated": ("1 if registry status is deprecated", "0/1", "clean_and_merge"),
    "published_at": ("Timestamp of the latest version", "UTC", "registry"),
    "updated_at": ("Timestamp of the last registry update", "UTC", "registry"),
    "n_versions": ("Published versions for this server", "count", "version history"),
    "first_published": ("Timestamp of the earliest version", "UTC", "version history"),
    "last_published": ("Timestamp of the most recent version", "UTC", "version history"),
    # distribution
    "n_packages": ("Package entries in the latest version", "count", "registry"),
    "package_types": ("Package registries used (semicolon separated)", "-", "registry"),
    "primary_package": ("First declared package registry, else 'none'", "-", "clean_and_merge"),
    "n_remotes": ("Remote endpoints declared", "count", "registry"),
    "remote_types": ("Transport types declared", "-", "registry"),
    "primary_transport": ("First declared transport, else 'none'", "-", "clean_and_merge"),
    "has_remote": ("1 if any remote endpoint is declared", "0/1", "clean_and_merge"),
    # code acquisition
    "code_status": ("Outcome of source fetch (ok/not_found/forbidden/...)", "-", "code fetch"),
    "has_code": ("1 if source files were fetched and scanned", "0/1", "clean_and_merge"),
    "n_files": ("Source files fetched for the repository", "count", "code fetch"),
    "bytes_fetched": ("Bytes of source fetched", "bytes", "code fetch"),
    # repository signals (RQ3)
    "stars": ("GitHub stargazer count", "count", "GitHub API"),
    "log_stars": ("log(1 + stars); stars span several orders of magnitude", "log count", "clean_and_merge"),
    "popularity_decile": ("Decile of star rank, 1 = least popular", "1-10", "feature engineering"),
    "is_popular": ("1 if stars >= 10", "0/1", "feature engineering"),
    "language": ("Primary repository language reported by GitHub", "-", "GitHub API"),
    "archived": ("1 if the repository is archived", "0/1", "GitHub API"),
    "is_archived": ("archived with NaN treated as 0", "0/1", "feature engineering"),
    "open_issues": ("Open issues on the repository", "count", "GitHub API"),
    "license": ("SPDX license identifier, blank if none", "-", "GitHub API"),
    "has_license": ("1 if a license is declared", "0/1", "feature engineering"),
    "days_since_push": ("Days since the last push to the repository", "days", "feature engineering"),
    "repo_age_days": ("Days since repository creation", "days", "feature engineering"),
    "is_maintained": ("1 if pushed within the last 180 days", "0/1", "feature engineering"),
    # scan outputs (RQ1, RQ5)
    "n_findings_total": ("Findings across all threat classes", "count", "scanner"),
    "n_classes_hit": ("Distinct threat classes with >=1 finding", "count", "feature engineering"),
    "severity_score": ("Sum of severity weights over classes present", "weighted", "feature engineering"),
    "has_high_severity": ("1 if any high-severity class is present", "0/1", "feature engineering"),
    "any_finding": ("1 if any finding was produced", "0/1", "feature engineering"),
    "findings_per_file": ("Findings divided by files scanned", "per file", "feature engineering"),
    "n_tools_found": ("Tool definitions recovered from source", "count", "scanner"),
    "mean_tool_desc_len": ("Mean characters per tool description", "chars", "feature engineering"),
    # drift (RQ2)
    "n_drift_events": ("Version-to-version change events detected", "count", "drift detection"),
    "n_silent_drift": ("Events where identifiers held but meaning or destination changed", "count", "drift detection"),
    "n_breaking_drift": ("Events withdrawing a package or endpoint", "count", "drift detection"),
    "has_silent_drift": ("1 if any silent-semantic drift event occurred", "0/1", "feature engineering"),
    "median_days_between": ("Median days between changed versions", "days", "feature engineering"),
}

SEV_NOTE = {"high": "direct path to compromise if reachable",
            "medium": "weakens isolation or confidentiality",
            "low": "supply-chain hygiene concern"}


def define(col: str):
    if col in D:
        return D[col]
    if col.startswith("n_") and col[2:] in config.THREAT_CLASSES:
        cls = col[2:]
        return (f"Findings for {config.THREAT_CLASSES[cls]['name']}",
                "count", "scanner")
    if col.startswith("has_") and col[4:] in config.THREAT_CLASSES:
        cls = col[4:]
        return (f"1 if any {config.THREAT_CLASSES[cls]['name']} finding",
                "0/1", "feature engineering")
    return None


def main():
    print("=" * 76)
    print(" CODEBOOK GENERATION")
    print("=" * 76)
    df = load_master(config.FILE_MASTER)

    rows, undocumented = [], []
    for col in df.columns:
        d = define(col)
        if d is None:
            undocumented.append(col)
            d = ("!!! UNDOCUMENTED — add to the D dictionary !!!", "?", "?")
        s = df[col]
        rows.append({
            "variable": col, "dtype": str(s.dtype), "definition": d[0],
            "unit": d[1], "source_stage": d[2],
            "non_null": int(s.notna().sum()),
            "coverage": round(float(s.notna().mean()), 4),
            "example": str(s.dropna().iloc[0])[:44] if s.notna().any() else "",
        })
    cb = pd.DataFrame(rows)

    lines = ["=" * 104,
             " CODEBOOK — master.csv variable dictionary",
             f" Snapshot : {config.SNAPSHOT_LABEL}",
             f" Generated: {datetime.now(timezone.utc).isoformat()}",
             f" Variables: {len(cb)}   Servers: {len(df):,}", "=" * 104,
             f" {'Variable':<30s} {'Type':<9s} {'Unit':<11s} {'Cover':>6s}  "
             f"Definition", "-" * 104]
    for _, r in cb.iterrows():
        lines.append(f" {r['variable']:<30s} {r['dtype']:<9s} "
                     f"{r['unit']:<11s} {r['coverage']:>6.1%}  "
                     f"{r['definition']}")
    lines += ["-" * 104,
              " Coverage note: finding columns are populated only for servers",
              " whose source was fetched; others are NaN by design so that",
              " prevalence denominators stay correct.", "=" * 104]

    save_csv(cb, config.CODEBOOK_DIR / "codebook.csv", "codebook")
    save_json(cb.to_dict("records"), config.CODEBOOK_DIR / "codebook.json",
              "codebook")
    save_txt("\n".join(lines), config.CODEBOOK_DIR / "codebook.txt", "codebook")

    # ---- threat catalogue as its own artifact -----------------------------
    tc = pd.DataFrame([{
        "class": cls,
        "name": spec["name"],
        "surface": spec["surface"],
        "severity": spec["severity"],
        "severity_weight": config.SEVERITY_WEIGHT[spec["severity"]],
        "n_patterns": len(spec["patterns"]),
        "rationale": SEV_NOTE[spec["severity"]],
    } for cls, spec in config.THREAT_CLASSES.items()])
    save_csv(tc, config.CODEBOOK_DIR / "threat_catalogue.csv",
             "threat catalogue")

    tl = ["=" * 92, " THREAT CATALOGUE (detection classes used by the scanner)",
          f" Snapshot: {config.SNAPSHOT_LABEL}", "=" * 92,
          f" {'Class':<28s} {'Severity':<9s} {'Surface':<10s} {'Pat':>4s}  Name",
          "-" * 92]
    for _, r in tc.iterrows():
        tl.append(f" {r['class']:<28s} {r['severity']:<9s} "
                  f"{r['surface']:<10s} {r['n_patterns']:>4d}  {r['name']}")
    tl += ["-" * 92,
           " Surfaces: tool = recovered tool definitions; code = handler",
           " source; manifest = package and build manifests.",
           " Detection is pattern-based, so findings are risk indicators",
           " rather than confirmed vulnerabilities; RQ4 reports precision",
           " and recall against manual labels.", "=" * 92]
    save_txt("\n".join(tl), config.CODEBOOK_DIR / "threat_catalogue.txt",
             "threat catalogue")

    if undocumented:
        print(f"  [WARN] {len(undocumented)} undocumented: {undocumented[:8]}")
        print("         add them to the D dictionary in this script")
    else:
        print(f"  [ OK ] all {len(cb)} variables documented")
    print("  Next: python labeling/01_sample_for_labeling.py")


if __name__ == "__main__":
    main()
