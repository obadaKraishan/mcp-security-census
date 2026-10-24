"""
============================================================================
 01_harvest_registry.py, Harvest the Full MCP Registry (population census)
============================================================================
 Project : Same Name, Different Server (MCP Security Census, IEEE BigData 2026)
 Purpose : Crawl every page of the official MCP Registry and store the raw
           server records plus a snapshot manifest. This defines the study
           population, so the snapshot date and counts are recorded exactly.
 Input   : https://registry.modelcontextprotocol.io/v0/servers (public)
 Output  : data/raw/registry_servers.parquet   (one row per version record)
           data/raw/registry_manifest.json     (machine-readable snapshot)
           data/raw/registry_harvest_report.txt(human-readable summary)
 Usage   : python data_collection/01_harvest_registry.py
           python data_collection/01_harvest_registry.py --max-pages 5
           python data_collection/01_harvest_registry.py --force
============================================================================
 Safety: an incomplete crawl NEVER overwrites a larger existing snapshot.
 Partial results are written to registry_servers_partial.parquet instead.
============================================================================
"""

import argparse
import json
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setup import config                                     # noqa: E402
from utils.io_helpers import save_json, save_txt             # noqa: E402

HEADERS = {"User-Agent": "mcp-security-census/1.0 (academic research)",
           "Accept": "application/json"}


def fetch_page(cursor: str | None) -> dict | None:
    """One page of /v0/servers with retries and exponential backoff."""
    url = f"{config.REGISTRY_SERVERS}?limit={config.REGISTRY_PAGE_SIZE}"
    if cursor:
        url += f"&cursor={urllib.parse.quote(cursor, safe='')}"
    for attempt in range(config.REGISTRY_MAX_RETRIES):
        try:
            r = requests.get(url, headers=HEADERS,
                             timeout=config.REGISTRY_TIMEOUT_S)
            if r.status_code == 429:
                wait = min(120, 5 * (2 ** attempt))
                print(f"\n    [RATE LIMIT] sleeping {wait}s")
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                wait = min(60, 5 * (2 ** attempt))
                print(f"\n    [WARN] HTTP {r.status_code}; retry "
                      f"{attempt+1}/{config.REGISTRY_MAX_RETRIES} in {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == config.REGISTRY_MAX_RETRIES - 1:
                print(f"\n    [FAIL] {type(e).__name__} after "
                      f"{config.REGISTRY_MAX_RETRIES} attempts: {e}")
                return None
            wait = min(60, 5 * (2 ** attempt))
            print(f"\n    [WARN] {type(e).__name__}; retry "
                  f"{attempt+1}/{config.REGISTRY_MAX_RETRIES} in {wait}s")
            time.sleep(wait)
    return None


def flatten(record: dict) -> dict:
    """Flatten one registry record into a tabular row.

    Shape (verified Aug 2026):
      {"server": {name, description, title, version, repository{}, packages[],
                  remotes[]},
       "_meta": {"io.modelcontextprotocol.registry/official":
                  {status, publishedAt, updatedAt, isLatest, ...}}}
    """
    srv = record.get("server", {}) or {}
    meta = ((record.get("_meta", {}) or {})
            .get("io.modelcontextprotocol.registry/official", {}) or {})
    repo = srv.get("repository", {}) or {}
    packages = srv.get("packages", []) or []
    remotes = srv.get("remotes", []) or []

    return {
        # identity
        "name": srv.get("name", ""),
        "version": srv.get("version", ""),
        "title": srv.get("title", ""),
        "description": srv.get("description", ""),
        "schema": srv.get("$schema", ""),
        # provenance
        "repo_url": repo.get("url", ""),
        "repo_source": repo.get("source", ""),
        # distribution
        "n_packages": len(packages),
        "package_types": ";".join(
            sorted({p.get("registryType") or p.get("registry_name") or "?"
                    for p in packages})),
        "package_ids": ";".join(
            str(p.get("identifier") or p.get("name") or "")[:80]
            for p in packages)[:500],
        "n_remotes": len(remotes),
        "remote_types": ";".join(sorted({r.get("type", "?") for r in remotes})),
        "remote_urls": ";".join(str(r.get("url", ""))[:120]
                                for r in remotes)[:500],
        # registry metadata
        "status": meta.get("status", ""),
        "published_at": meta.get("publishedAt", ""),
        "updated_at": meta.get("updatedAt", ""),
        "is_latest": bool(meta.get("isLatest", False)),
        # keep the raw payload so later stages can reread without refetching
        "raw_json": json.dumps(record, separators=(",", ":")),
    }


def main():
    ap = argparse.ArgumentParser(description="Harvest the MCP registry")
    ap.add_argument("--max-pages", type=int, default=None,
                    help="stop after N pages (smoke test)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing snapshot even if this crawl "
                         "is smaller or incomplete")
    args = ap.parse_args()

    print("=" * 76)
    print(" STEP 1/3 (COLLECTION) — HARVEST MCP REGISTRY")
    print("=" * 76)
    print(f"  Endpoint : {config.REGISTRY_SERVERS}")
    print(f"  Page size: {config.REGISTRY_PAGE_SIZE}")
    print(f"  Timeout  : {config.REGISTRY_TIMEOUT_S}s, "
          f"{config.REGISTRY_MAX_RETRIES} retries")
    print(f"  Snapshot : {config.SNAPSHOT_LABEL}")
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)

    rows, cursor, pages, t0 = [], None, 0, time.time()
    while True:
        data = fetch_page(cursor)
        if data is None:
            print("\n  [WARN] page fetch failed; stopping early")
            break
        servers = data.get("servers", [])
        if not servers:
            break
        rows.extend(flatten(s) for s in servers)
        pages += 1
        cursor = (data.get("metadata", {}) or {}).get("nextCursor")
        rate = len(rows) / max(time.time() - t0, 1e-6)
        print(f"\r  page {pages:>4}  records {len(rows):>7,}  "
              f"({rate:.0f} rec/s)", end="", flush=True)
        if not cursor or (args.max_pages and pages >= args.max_pages):
            break
        time.sleep(config.REGISTRY_PAGE_DELAY_S)
    elapsed = time.time() - t0
    print()

    if not rows:
        print("  [FAIL] no records harvested — existing snapshot untouched")
        sys.exit(1)

    # ---- snapshot safety: never clobber a better snapshot -------------------
    complete = (cursor is None) or bool(args.max_pages)
    if not complete:
        print(f"  [WARN] INCOMPLETE crawl: stopped after page {pages}")
    df = pd.DataFrame(rows)

    if config.FILE_REGISTRY_RAW.exists() and not args.force:
        prev_n = len(pd.read_parquet(config.FILE_REGISTRY_RAW))
        if len(df) < prev_n and not complete:
            alt = config.DATA_RAW / "registry_servers_partial.parquet"
            df.to_parquet(alt, index=False)
            print(f"  [KEPT ] existing snapshot has {prev_n:,} records, this "
                  f"crawl only {len(df):,}")
            print(f"          partial crawl saved separately -> {alt}")
            print("          re-run to try again, or use --force to overwrite")
            sys.exit(1)
    df.to_parquet(config.FILE_REGISTRY_RAW, index=False)

    # ---- population summary ------------------------------------------------
    n_records = len(df)
    n_unique = df["name"].nunique()
    n_latest = int(df["is_latest"].sum())
    with_repo = int((df["repo_url"] != "").sum())
    with_pkg = int((df["n_packages"] > 0).sum())
    with_remote = int((df["n_remotes"] > 0).sum())
    status_counts = df["status"].value_counts().to_dict()
    pkg_counts = (df.loc[df["package_types"] != "", "package_types"]
                  .str.split(";").explode().value_counts().head(8).to_dict())
    remote_counts = (df.loc[df["remote_types"] != "", "remote_types"]
                     .str.split(";").explode().value_counts().to_dict())
    dates = pd.to_datetime(df["published_at"], errors="coerce", utc=True,
                           format="mixed")

    manifest = {
        "snapshot_label": config.SNAPSHOT_LABEL,
        "harvested_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": config.REGISTRY_SERVERS,
        "crawl_complete": bool(complete),
        "pages": pages,
        "version_records": n_records,
        "unique_servers": n_unique,
        "latest_version_records": n_latest,
        "with_repo_url": with_repo,
        "with_package": with_pkg,
        "with_remote": with_remote,
        "status_counts": status_counts,
        "package_type_counts": pkg_counts,
        "remote_type_counts": remote_counts,
        "published_min": str(dates.min()),
        "published_max": str(dates.max()),
        "elapsed_seconds": round(elapsed, 1),
        "file": str(config.FILE_REGISTRY_RAW),
    }
    save_json(manifest, config.FILE_REGISTRY_MANIFEST, "registry manifest")

    lines = [
        "=" * 76, " MCP REGISTRY HARVEST REPORT",
        f" Generated : {manifest['harvested_at_utc']}",
        f" Snapshot  : {config.SNAPSHOT_LABEL}",
        f" Complete  : {'yes' if complete else 'NO (partial crawl)'}",
        "=" * 76,
        f" Pages crawled          : {pages:,}",
        f" Version records        : {n_records:,}",
        f" Unique servers         : {n_unique:,}   <- study population",
        f" Latest-version records : {n_latest:,}",
        "-" * 76,
        f" With repository URL    : {with_repo:,} ({with_repo/n_records:.1%})"
        "   <- eligible for code analysis",
        f" With a package         : {with_pkg:,} ({with_pkg/n_records:.1%})",
        f" With a remote endpoint : {with_remote:,} ({with_remote/n_records:.1%})",
        f" Published range        : {str(dates.min())[:10]} .. "
        f"{str(dates.max())[:10]}",
        f" Elapsed                : {elapsed/60:.1f} min",
        "-" * 76, " Status counts:",
    ]
    for k, v in status_counts.items():
        lines.append(f"   {k:<24s} {v:>8,}")
    lines.append(" Package types:")
    for k, v in pkg_counts.items():
        lines.append(f"   {k:<24s} {v:>8,}")
    lines.append(" Remote transport types:")
    for k, v in remote_counts.items():
        lines.append(f"   {k:<24s} {v:>8,}")
    lines.append("=" * 76)
    save_txt("\n".join(lines), config.DATA_RAW / "registry_harvest_report.txt",
             "harvest report")

    print("\n".join(lines))
    print(f"  Parquet : {config.FILE_REGISTRY_RAW}")
    if not complete:
        print("  [ACTION] crawl was partial — re-run to complete the snapshot")
    print("  Next    : python data_collection/02_fetch_version_history.py")


if __name__ == "__main__":
    main()