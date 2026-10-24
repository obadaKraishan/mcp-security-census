"""
============================================================================
 02_fetch_version_history.py, Per-Server Version History (feeds RQ2 drift)
============================================================================
 Project : Same Name, Different Server (MCP Security Census, IEEE BigData 2026)
 Purpose : For every unique server, pull the full list of published versions
           and capture, per version, everything the registry actually
           versions: description, package identifiers and versions, and
           remote endpoint URLs. RQ2 compares consecutive versions to detect
           changes an already-installed client would not be told about.
 Input   : data/raw/registry_servers.parquet
 Output  : data/raw/server_versions.parquet
           data/raw/version_history_report.txt (+ .json summary)
 Usage   : python data_collection/02_fetch_version_history.py
           python data_collection/02_fetch_version_history.py --max-servers 50
           python data_collection/02_fetch_version_history.py --multi-only
============================================================================
 Every response is cached to disk, so re-runs are fast and resumable.
============================================================================
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setup import config                                   # noqa: E402
from utils.io_helpers import save_json, save_txt           # noqa: E402

HEADERS = {"User-Agent": "mcp-security-census/1.0 (academic research)",
           "Accept": "application/json"}
CHECKPOINT_EVERY = 1000


def cached_get(url: str) -> dict | None:
    """GET with on-disk cache; returns parsed JSON or None."""
    key = hashlib.sha1(url.encode()).hexdigest()
    cache_file = config.CACHE_DIR / f"ver_{key}.json"
    if cache_file.exists():
        try:
            txt = cache_file.read_text()
            return None if txt == "null" else json.loads(txt)
        except json.JSONDecodeError:
            cache_file.unlink()

    for attempt in range(config.REGISTRY_MAX_RETRIES):
        try:
            r = requests.get(url, headers=HEADERS,
                             timeout=config.REGISTRY_TIMEOUT_S)
            if r.status_code == 429:
                wait = min(120, 5 * (2 ** attempt))
                print(f"\n    [RATE LIMIT] sleeping {wait}s")
                time.sleep(wait)
                continue
            if r.status_code in (404, 410):
                cache_file.write_text("null")
                return None
            r.raise_for_status()
            data = r.json()
            cache_file.write_text(json.dumps(data))
            return data
        except requests.RequestException:
            if attempt == config.REGISTRY_MAX_RETRIES - 1:
                return None
            time.sleep(min(60, 5 * (2 ** attempt)))
    return None


def extract_tools(payload: dict) -> list:
    """Recover tool definitions if a record ever embeds them.

    Observed to be absent registry-wide (reported as a finding), but the
    extraction is kept so the pipeline benefits if the schema gains them.
    """
    srv = payload.get("server", payload) or {}
    tools = srv.get("tools") or (srv.get("capabilities", {}) or {}).get("tools") or []
    if isinstance(tools, dict):
        tools = [{"name": k, **(v if isinstance(v, dict) else {})}
                 for k, v in tools.items()]
    out = []
    for t in tools if isinstance(tools, list) else []:
        if isinstance(t, dict):
            out.append({"name": t.get("name", ""),
                        "description": (t.get("description") or "")[:2000]})
    return out


def summarize_packages(srv: dict) -> dict:
    """Per-version package fingerprint (what a client would install)."""
    pkgs = srv.get("packages") or []
    ids, types, vers, runtimes = [], [], [], []
    for p in pkgs:
        ids.append(str(p.get("identifier") or p.get("name") or ""))
        types.append(str(p.get("registryType") or p.get("registry_name") or ""))
        vers.append(str(p.get("version") or ""))
        # runtime hints carry execution surface (docker args, runtime hint)
        rt = p.get("runtimeHint") or p.get("runtime_hint") or ""
        if rt:
            runtimes.append(str(rt))
    return {
        "n_packages": len(pkgs),
        "package_ids": "|".join(sorted(ids))[:800],
        "package_types": "|".join(sorted(set(types)))[:200],
        "package_versions": "|".join(sorted(vers))[:400],
        "runtime_hints": "|".join(sorted(set(runtimes)))[:200],
    }


def summarize_remotes(srv: dict) -> dict:
    """Per-version remote fingerprint (where a client would connect)."""
    remotes = srv.get("remotes") or []
    urls, types, hosts = [], [], []
    for r in remotes:
        u = str(r.get("url") or "")
        urls.append(u)
        types.append(str(r.get("type") or ""))
        try:
            hosts.append(urllib.parse.urlparse(u).netloc.lower())
        except ValueError:
            pass
    return {
        "n_remotes": len(remotes),
        "remote_urls": "|".join(sorted(urls))[:800],
        "remote_types": "|".join(sorted(set(types)))[:200],
        "remote_hosts": "|".join(sorted(set(h for h in hosts if h)))[:400],
    }


def main():
    ap = argparse.ArgumentParser(description="Fetch per-server version history")
    ap.add_argument("--max-servers", type=int, default=None)
    ap.add_argument("--multi-only", action="store_true",
                    help="only servers that already show >1 version record")
    args = ap.parse_args()

    print("=" * 76)
    print(" STEP 2/3 (COLLECTION) — VERSION HISTORY (cached, resumable)")
    print("=" * 76)
    if not config.FILE_REGISTRY_RAW.exists():
        sys.exit("[FAIL] registry_servers.parquet missing — run 01 first")
    reg = pd.read_parquet(config.FILE_REGISTRY_RAW)
    counts = reg.groupby("name").size()
    names = sorted(counts.index)
    print(f"  unique servers in snapshot : {len(names):,}")
    if args.multi_only:
        names = sorted(counts[counts > 1].index)
        print(f"  --multi-only filter        : {len(names):,} servers")
    if args.max_servers:
        names = names[:args.max_servers]
        print(f"  --max-servers cap          : {len(names):,} servers")
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    rows, n_fail, t0 = [], 0, time.time()
    for i, name in enumerate(tqdm(names, desc="  fetching", unit="srv")):
        url = (f"{config.REGISTRY_SERVERS}/"
               f"{urllib.parse.quote(name, safe='')}/versions")
        data = cached_get(url)
        if data is None:
            n_fail += 1
            continue
        versions = data.get("servers") or data.get("versions") or []
        for v in versions:
            srv = v.get("server", v) or {}
            meta = ((v.get("_meta", {}) or {})
                    .get("io.modelcontextprotocol.registry/official", {}) or {})
            tools = extract_tools(v)
            row = {
                "name": name,
                "version": srv.get("version", ""),
                "description": (srv.get("description") or "")[:2000],
                "title": (srv.get("title") or "")[:300],
                "repo_url": (srv.get("repository", {}) or {}).get("url", ""),
                "published_at": meta.get("publishedAt", ""),
                "updated_at": meta.get("updatedAt", ""),
                "is_latest": bool(meta.get("isLatest", False)),
                "status": meta.get("status", ""),
                "n_tools": len(tools),
                "tools_json": json.dumps(tools, separators=(",", ":"))[:20000],
            }
            row.update(summarize_packages(srv))
            row.update(summarize_remotes(srv))
            rows.append(row)
        if (i + 1) % CHECKPOINT_EVERY == 0 and rows:
            pd.DataFrame(rows).to_parquet(config.FILE_VERSIONS, index=False)

    if not rows:
        sys.exit("[FAIL] no version records collected")
    df = pd.DataFrame(rows)
    df.to_parquet(config.FILE_VERSIONS, index=False)

    per_server = df.groupby("name").size()
    multi = int((per_server > 1).sum())
    elapsed = time.time() - t0

    summary = {
        "snapshot_label": config.SNAPSHOT_LABEL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "servers_queried": len(names),
        "servers_with_history": int(per_server.size),
        "version_records": int(len(df)),
        "servers_multi_version": multi,
        "max_versions_single_server": int(per_server.max()),
        "median_versions": float(per_server.median()),
        "records_with_packages": int((df["n_packages"] > 0).sum()),
        "records_with_remotes": int((df["n_remotes"] > 0).sum()),
        "records_with_tools": int((df["n_tools"] > 0).sum()),
        "fetch_failures": n_fail,
        "elapsed_seconds": round(elapsed, 1),
    }
    save_json(summary, config.DATA_RAW / "version_history_summary.json",
              "version summary")

    lines = [
        "=" * 76, " VERSION HISTORY REPORT",
        f" Generated : {summary['generated_at_utc']}", "=" * 76,
        f" Servers queried             : {len(names):,}",
        f" Servers with history        : {per_server.size:,}",
        f" Total version records       : {len(df):,}",
        f" Servers with >1 version     : {multi:,} "
        f"({multi/max(per_server.size,1):.1%})   <- RQ2 analysis set",
        f" Max versions (one server)   : {int(per_server.max()):,}",
        f" Median versions per server  : {per_server.median():.1f}",
        "-" * 76, " Per-version fields captured for drift analysis:",
        f"   records with packages     : {summary['records_with_packages']:,} "
        f"({summary['records_with_packages']/len(df):.1%})",
        f"   records with remotes      : {summary['records_with_remotes']:,} "
        f"({summary['records_with_remotes']/len(df):.1%})",
        f"   records with tool defs    : {summary['records_with_tools']:,} "
        f"({summary['records_with_tools']/len(df):.1%})",
        f" Fetch failures              : {n_fail:,}",
        f" Elapsed                     : {elapsed/60:.1f} min",
        "-" * 76,
        " NOTE: the registry does not embed tool definitions in server",
        " records. Drift is therefore measured on what it does version:",
        " description text, package identifiers and versions, and remote",
        " endpoint URLs. That absence is itself reported as a finding.",
        "=" * 76,
    ]
    save_txt("\n".join(lines), config.DATA_RAW / "version_history_report.txt",
             "version report")
    print("\n".join(lines))
    print(f"  Parquet : {config.FILE_VERSIONS}")
    print("  Next    : python data_processing/03_detect_drift.py")


if __name__ == "__main__":
    main()
