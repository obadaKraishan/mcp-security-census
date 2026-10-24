"""
============================================================================
 02_scan_security.py, The Scanner (core contribution)
============================================================================
 Project : Same Name, Different Server (MCP Security Census, IEEE BigData 2026)
 Purpose : Apply the threat catalogue to every fetched server file and to the
           tool definitions recovered from that code. Emits one row per
           finding (server, class, severity, file, line, evidence) plus a
           per-server rollup. The evidence snippet is what makes RQ4's
           manual labelling possible.
 Input   : data/interim/servers_merged.csv, data/interim/code_index.csv,
           data/.code_cache/**
 Output  : data/interim/security_findings.csv (+ .json + scan_report.txt)
           data/interim/tool_definitions.csv   (recovered tool defs, RQ2/RQ1)
 Usage   : python data_processing/02_scan_security.py
           python data_processing/02_scan_security.py --max-servers 200
============================================================================
 Detection is pattern-based on source text. Findings are RISK INDICATORS,
 not confirmed vulnerabilities; RQ4 measures precision and recall.
============================================================================
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setup import config                                    # noqa: E402
from utils.io_helpers import save_csv, save_json, save_txt  # noqa: E402

# ---------------------------------------------------------------------------
# Tool-definition extraction (idioms verified against real MCP servers)
#   Python  : Tool(name="x", description="...")   and  @mcp.tool()/@server.tool()
#   TypeScript: server.registerTool("x", { description: "..." })
#               server.tool("x", "description", ...)
# ---------------------------------------------------------------------------
TOOL_PATTERNS = [
    # Python SDK: Tool(name="fetch", description="""...""")
    re.compile(r"""Tool\s*\(\s*name\s*=\s*["']([\w.\-]+)["']\s*,\s*"""
               r"""description\s*=\s*(?:"{3}|'{3}|["'])(.{0,3000}?)(?:"{3}|'{3}|["'])""",
               re.DOTALL),
    # TS SDK: server.registerTool("name", { title: ..., description: "..." })
    re.compile(r"""registerTool\s*\(\s*["']([\w.\-]+)["']\s*,\s*\{[^}]{0,400}?"""
               r"""description\s*:\s*["'`](.{0,3000}?)["'`]""", re.DOTALL),
    # TS SDK (older): server.tool("name", "description", ...)
    re.compile(r"""\.tool\s*\(\s*["']([\w.\-]+)["']\s*,\s*["'`](.{0,3000}?)["'`]""",
               re.DOTALL),
    # FastMCP decorator followed by a docstring
    re.compile(r"""@(?:mcp|server|app)\.tool\s*\([^)]*\)\s*\n\s*(?:async\s+)?def\s+"""
               r"""([\w]+)\s*\([^)]*\)[^:]*:\s*\n\s*(?:"{3}|'{3})(.{0,3000}?)(?:"{3}|'{3})""",
               re.DOTALL),
    # Python SDK with an enum/constant name: Tool(name=GitTools.STATUS,
    # description="...")  -- common in production servers
    re.compile(r"""Tool\s*\(\s*name\s*=\s*([A-Za-z_][\w.]*)\s*,\s*"""
               r"""description\s*=\s*(?:"{3}|'{3}|["'])(.{0,3000}?)(?:"{3}|'{3}|["'])""",
               re.DOTALL),
]

MAX_EVIDENCE = 220        # characters of context stored per finding
MAX_FINDINGS_PER_CLASS = 25   # cap per server per class (avoid file spam)


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None


def extract_tools(text: str) -> list:
    """Recover tool definitions (name + description) from source text."""
    found, seen = [], set()
    for rx in TOOL_PATTERNS:
        for m in rx.finditer(text):
            name = m.group(1).strip()
            desc = re.sub(r"\s+", " ", m.group(2)).strip()
            if name and name not in seen and len(name) < 80:
                seen.add(name)
                found.append({"tool_name": name, "tool_description": desc[:2000]})
    return found


def scan_text(text: str, surface: str) -> list:
    """Return (class, line_no, evidence) for every catalogue hit on `text`."""
    hits = []
    lines = text.split("\n")
    for cls, spec in config.THREAT_CLASSES.items():
        if spec["surface"] != surface:
            continue
        per_class = 0
        for rx in config.COMPILED_THREATS[cls]:
            for m in rx.finditer(text):
                if per_class >= MAX_FINDINGS_PER_CLASS:
                    break
                line_no = text.count("\n", 0, m.start()) + 1
                raw = lines[line_no - 1] if line_no <= len(lines) else m.group(0)
                evidence = re.sub(r"\s+", " ", raw).strip()[:MAX_EVIDENCE]
                hits.append({"threat_class": cls, "line": line_no,
                             "evidence": evidence,
                             "severity": spec["severity"],
                             "surface": surface})
                per_class += 1
    return hits


def main():
    ap = argparse.ArgumentParser(description="Scan fetched server code")
    ap.add_argument("--max-servers", type=int, default=None)
    args = ap.parse_args()

    print("=" * 76)
    print(" STEP 2/4 (PROCESSING) — SECURITY SCAN")
    print("=" * 76)
    if not config.FILE_SERVERS_MERGED.exists():
        sys.exit("[FAIL] servers_merged.csv missing — run processing 01")
    if not config.FILE_CODE_INDEX.exists():
        sys.exit("[FAIL] code_index.csv missing — run collection 03")

    servers = pd.read_csv(config.FILE_SERVERS_MERGED, low_memory=False)
    idx = pd.read_csv(config.FILE_CODE_INDEX, low_memory=False)
    idx = idx[idx["status"] == "ok"]
    print(f"  servers in frame     : {len(servers):,}")
    print(f"  fetched files        : {len(idx):,}")

    # map repo -> the servers published from it
    repo_servers = defaultdict(list)
    for _, s in servers[servers["has_repo"] == True].iterrows():   # noqa: E712
        repo_servers[(s["owner"], s["repo"])].append(s["name"])

    repos = list(idx.groupby(["owner", "repo"]))
    if args.max_servers:
        repos = repos[:args.max_servers]
    print(f"  repositories to scan : {len(repos):,}")

    findings, tools, scanned_files, scanned_bytes = [], [], 0, 0
    repos_scanned = 0
    n_skipped_test = 0

    for (owner, repo), grp in tqdm(repos, desc="  scanning", unit="repo"):
        names = repo_servers.get((owner, repo), [f"{owner}/{repo}"])
        repo_hits, repo_tools = [], []
        for _, f in grp.iterrows():
            lp = Path(str(f.get("local_path", "")))
            if not lp.exists():
                continue
            text = read_text(lp)
            if text is None:
                continue
            scanned_files += 1
            scanned_bytes += len(text)
            base = lp.name.split("__")[-1]
            # Skip test, spec, demo, and fixture files: credentials and unsafe
            # calls there are non-operational and inflate every class.
            low = str(f["path"]).lower()
            if re.search(r"(^|[/_.\-])(test|tests|spec|specs|demo|fixture|"
                         r"fixtures|mock|mocks|example|examples|sample|"
                         r"samples|conformance|benchmark)([/_.\-]|$)", low):
                n_skipped_test += 1
                continue
            is_manifest = base in config.MANIFEST_FILES
            surface = "manifest" if is_manifest else "code"

            # 1. code / manifest surfaces
            for h in scan_text(text, surface):
                h.update({"file": f["path"]})
                repo_hits.append(h)

            # 2. tool definitions -> tool surface (poisoning checks)
            for t in extract_tools(text):
                t.update({"file": f["path"]})
                repo_tools.append(t)
                blob = f"{t['tool_name']} {t['tool_description']}"
                for h in scan_text(blob, "tool"):
                    h.update({"file": f["path"],
                              "tool_name": t["tool_name"],
                              "evidence": t["tool_description"][:MAX_EVIDENCE]})
                    repo_hits.append(h)

        repos_scanned += 1
        for name in names:
            for h in repo_hits:
                findings.append({"server_name": name, "owner": owner,
                                 "repo": repo, **h})
            for t in repo_tools:
                tools.append({"server_name": name, "owner": owner,
                              "repo": repo, **t})

    if not findings:
        print("  [WARN] no findings produced")
    fdf = pd.DataFrame(findings)
    tdf = pd.DataFrame(tools)
    save_csv(fdf, config.FILE_FINDINGS, "security findings")
    save_csv(tdf, config.DATA_INTERIM / "tool_definitions.csv",
             "tool definitions")

    # ---- report -----------------------------------------------------------
    n_srv_scanned = fdf["server_name"].nunique() if len(fdf) else 0
    by_class = (fdf.groupby("threat_class")["server_name"].nunique()
                if len(fdf) else pd.Series(dtype=int))
    by_sev = (fdf.drop_duplicates(["server_name", "threat_class"])
              .groupby("severity")["server_name"].nunique()
              if len(fdf) else pd.Series(dtype=int))

    lines = [
        "=" * 76, " SECURITY SCAN REPORT",
        f" Generated : {datetime.now(timezone.utc).isoformat()}",
        f" Snapshot  : {config.SNAPSHOT_LABEL}", "=" * 76,
        f" Repositories scanned    : {repos_scanned:,}",
        f" Files read              : {scanned_files:,}",
        f" Test/demo files skipped : {n_skipped_test:,}",
        f" Text analysed           : {scanned_bytes/1e6:.1f} MB",
        f" Tool definitions found  : {len(tdf):,} "
        f"({tdf['repo'].nunique() if len(tdf) else 0:,} repos)",
        f" Findings emitted        : {len(fdf):,}",
        f" Servers with >=1 finding: {n_srv_scanned:,}",
        "-" * 76, " Servers affected per threat class:",
    ]
    for cls in config.THREAT_CLASSES:
        n = int(by_class.get(cls, 0))
        lines.append(f"   {cls:<30s} {n:>8,}")
    lines.append(" Servers affected per severity:")
    for sev in ["high", "medium", "low"]:
        lines.append(f"   {sev:<30s} {int(by_sev.get(sev, 0)):>8,}")
    lines += ["-" * 76,
              " Findings are pattern-based risk indicators, not confirmed",
              " vulnerabilities. RQ4 reports precision and recall against a",
              " manually labelled sample.", "=" * 76]

    save_json({"repos_scanned": repos_scanned, "files_read": scanned_files,
               "findings": int(len(fdf)), "tool_definitions": int(len(tdf)),
               "servers_by_class": {k: int(v) for k, v in by_class.items()},
               "generated_at_utc": datetime.now(timezone.utc).isoformat()},
              config.DATA_INTERIM / "scan_summary.json", "scan summary")
    save_txt("\n".join(lines), config.DATA_INTERIM / "scan_report.txt",
             "scan report")
    print("\n".join(lines))
    print("  Next: python data_processing/03_detect_drift.py")


if __name__ == "__main__":
    main()
