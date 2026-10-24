"""
============================================================================
 03_fetch_server_code.py, Fetch Source Code for Repository-Linked Servers
============================================================================
 Project : Same Name, Different Server (MCP Security Census, IEEE BigData 2026)
 Purpose : Download a bounded, relevant slice of each server's source code so
           the scanner (data_processing/02) can inspect handler code, and so
           tool definitions can be recovered from code. This matters because
           the registry itself does not embed tool definitions: code is the
           only place where tool names, descriptions, and schemas live.
 Input   : data/raw/registry_servers.parquet (repo_url column)
 Output  : data/.code_cache/<owner>__<repo>/...   (fetched files)
           data/interim/code_index.csv           (one row per fetched file)
           data/interim/code_fetch_report.txt (+ .json summary)
 Usage   : python data_collection/03_fetch_server_code.py
           python data_collection/03_fetch_server_code.py --max-repos 25
           python data_collection/03_fetch_server_code.py --workers 8
============================================================================
 Budget: ~2 API calls per repo (metadata + tree) plus raw file downloads,
 which do not count against the API rate limit. Fully cached and resumable.
 Failure reasons are recorded separately (not_found / forbidden /
 dmca_blocked / empty_repo / unreachable) so attrition can be reported.
============================================================================
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setup import config                                    # noqa: E402
from utils.io_helpers import save_json, save_txt            # noqa: E402

load_dotenv(config.PROJECT_ROOT / ".env")
TOKEN = os.getenv("GITHUB_TOKEN", "")
HEADERS = {"Accept": "application/vnd.github+json",
           "User-Agent": "mcp-security-census/1.0"}
if TOKEN:
    HEADERS["Authorization"] = f"Bearer {TOKEN}"

RAW_BASE = "https://raw.githubusercontent.com"

# Files most likely to contain tool definitions and handler logic
PRIORITY_HINTS = ("server", "index", "main", "tool", "handler", "app", "mcp",
                  "src/", "__init__")

# Directories that waste the per-repo file budget in monorepos
SKIP_DIRS = ("node_modules/", "/dist/", "/build/", "/vendor/", "/.venv/",
             "/test", "/tests/", "/__tests__/", "/spec/", "/examples/",
             "/example/", "/templates/", "/template/", "/fixtures/",
             "/docs/", "/doc/", "/resources/packs/", "/site-packages/",
             "/.github/", "/coverage/", "/.next/", "/target/")

# HTTP status -> recorded reason (kept distinct for attrition reporting)
STATUS_REASON = {404: "not_found", 403: "forbidden", 451: "dmca_blocked",
                 409: "empty_repo"}

_print_lock = threading.Lock()


def parse_repo(url: str):
    """https://github.com/owner/repo(.git) -> ('owner', 'repo')"""
    m = re.match(r"https?://github\.com/([^/]+)/([^/\s#?]+)", str(url).strip())
    if not m:
        return None
    return m.group(1), m.group(2).replace(".git", "")


def api_get(url, params=None):
    """GitHub API GET. Returns dict, or {'__error__': code} for hard errors,
    or None if the request never succeeded."""
    for attempt in range(config.GITHUB_MAX_RETRIES):
        try:
            r = requests.get(url, headers=HEADERS, params=params,
                             timeout=config.GITHUB_TIMEOUT_S)
            # true rate-limit exhaustion: wait for the reset
            if r.status_code == 403 and \
               r.headers.get("X-RateLimit-Remaining") == "0":
                reset = int(r.headers.get("X-RateLimit-Reset",
                                          time.time() + 60))
                wait = max(reset - time.time(), 0) + 5
                with _print_lock:
                    print(f"\n    [RATE LIMIT] sleeping {wait/60:.1f} min")
                time.sleep(wait)
                continue
            if r.status_code in STATUS_REASON:
                return {"__error__": r.status_code}
            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == config.GITHUB_MAX_RETRIES - 1:
                return None
            time.sleep(2 ** attempt)
    return None


def score_path(path: str) -> int:
    """Rank files so root manifests and likely MCP server code come first."""
    base = path.split("/")[-1]
    depth = path.count("/")
    if base in config.MANIFEST_FILES:
        return 0 if depth == 0 else 4 + min(depth, 4)
    hint = 0 if any(h in path.lower() for h in PRIORITY_HINTS) else 3
    return 1 + hint + min(depth, 4)


def _fail_row(owner, repo, status, meta=None):
    meta = meta or {}
    return {"owner": owner, "repo": repo, "status": status, "path": "",
            "bytes": 0, "local_path": "",
            "stars": meta.get("stargazers_count"),
            "pushed_at": meta.get("pushed_at", ""),
            "created_at": meta.get("created_at", ""),
            "language": meta.get("language", ""),
            "archived": meta.get("archived"),
            "open_issues": meta.get("open_issues_count"),
            "license": meta.get("license")}


def fetch_repo(owner: str, repo: str) -> list:
    """Fetch repo metadata + a bounded set of files. Returns index rows."""
    out_dir = config.CODE_CACHE / f"{owner}__{repo}"
    meta_file = out_dir / "_repo_meta.json"

    # ---- 1. repository metadata (cached) ----------------------------------
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text())
        except json.JSONDecodeError:
            meta_file.unlink()
            meta = None
    else:
        meta = None

    if meta is None:
        raw = api_get(f"{config.GITHUB_API_BASE}/repos/{owner}/{repo}")
        if raw is None:
            return [_fail_row(owner, repo, "unreachable")]
        if "__error__" in raw:
            return [_fail_row(owner, repo,
                              STATUS_REASON.get(raw["__error__"], "unreachable"))]
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "stargazers_count": raw.get("stargazers_count"),
            "forks_count": raw.get("forks_count"),
            "pushed_at": raw.get("pushed_at"),
            "created_at": raw.get("created_at"),
            "language": raw.get("language"),
            "archived": raw.get("archived"),
            "open_issues_count": raw.get("open_issues_count"),
            "default_branch": raw.get("default_branch", "main"),
            "license": (raw.get("license") or {}).get("spdx_id"),
            "size_kb": raw.get("size"),
        }
        meta_file.write_text(json.dumps(meta))

    branch = meta.get("default_branch") or "main"

    # ---- 2. file tree (cached) -------------------------------------------
    tree_file = out_dir / "_tree.json"
    if tree_file.exists():
        try:
            tree = json.loads(tree_file.read_text())
        except json.JSONDecodeError:
            tree_file.unlink()
            tree = None
    else:
        tree = None

    if tree is None:
        tree = api_get(f"{config.GITHUB_API_BASE}/repos/{owner}/{repo}/"
                       f"git/trees/{branch}", params={"recursive": "1"})
        if tree is None or "__error__" in tree:
            return [_fail_row(owner, repo, "tree_unavailable", meta)]
        out_dir.mkdir(parents=True, exist_ok=True)
        tree_file.write_text(json.dumps(
            {"tree": [{"path": n.get("path"), "type": n.get("type"),
                       "size": n.get("size")} for n in tree.get("tree", [])],
             "truncated": tree.get("truncated", False)}))
        tree = json.loads(tree_file.read_text())

    # ---- 3. choose files --------------------------------------------------
    cands = []
    for node in tree.get("tree", []):
        if node.get("type") != "blob":
            continue
        p = node.get("path", "") or ""
        size = node.get("size", 0) or 0
        ext = Path(p).suffix.lower()
        base = p.split("/")[-1]
        if base not in config.MANIFEST_FILES and ext not in config.CODE_EXTENSIONS:
            continue
        if not (0 < size <= config.MAX_FILE_BYTES):
            continue
        lower = "/" + p.lower()
        if any(s in lower for s in SKIP_DIRS):
            continue
        cands.append((score_path(p), p, size))
    cands.sort()
    cands = cands[:config.MAX_FILES_PER_REPO]

    # ---- 4. download the chosen files (raw.githubusercontent, no API cost)
    rows = []
    for _, path, _size in cands:
        local = out_dir / path.replace("/", "__")
        if not local.exists():
            try:
                r = requests.get(f"{RAW_BASE}/{owner}/{repo}/{branch}/{path}",
                                 headers={"User-Agent": HEADERS["User-Agent"]},
                                 timeout=config.GITHUB_TIMEOUT_S)
                if r.status_code != 200:
                    continue
                out_dir.mkdir(parents=True, exist_ok=True)
                local.write_bytes(r.content[:config.MAX_FILE_BYTES])
            except requests.RequestException:
                continue
        rows.append({"owner": owner, "repo": repo, "status": "ok",
                     "path": path, "bytes": local.stat().st_size,
                     "local_path": str(local),
                     "stars": meta.get("stargazers_count"),
                     "pushed_at": meta.get("pushed_at", ""),
                     "created_at": meta.get("created_at", ""),
                     "language": meta.get("language", ""),
                     "archived": meta.get("archived"),
                     "open_issues": meta.get("open_issues_count"),
                     "license": meta.get("license")})

    return rows or [_fail_row(owner, repo, "no_files", meta)]


def main():
    ap = argparse.ArgumentParser(description="Fetch server source code")
    ap.add_argument("--max-repos", type=int, default=None,
                    help="cap repos processed this run")
    ap.add_argument("--workers", type=int, default=6,
                    help="parallel fetch threads (default 6)")
    args = ap.parse_args()

    print("=" * 76)
    print(" STEP 3/3 (COLLECTION) — SERVER SOURCE CODE (cached, resumable)")
    print("=" * 76)
    if not TOKEN:
        sys.exit("[FAIL] GITHUB_TOKEN missing — run setup_environment.py")
    if not config.FILE_REGISTRY_RAW.exists():
        sys.exit("[FAIL] registry_servers.parquet missing — run 01 first")

    reg = pd.read_parquet(config.FILE_REGISTRY_RAW)
    latest = reg[reg["is_latest"]] if reg["is_latest"].any() else reg
    repos = {}
    for _, r in latest.iterrows():
        pr = parse_repo(r["repo_url"])
        if pr:
            repos.setdefault(pr, []).append(r["name"])
    print(f"  servers in snapshot      : {latest['name'].nunique():,}")
    print(f"  distinct GitHub repos    : {len(repos):,}")

    done = set()
    if config.FILE_CODE_INDEX.exists():
        prev = pd.read_csv(config.FILE_CODE_INDEX)
        done = set(zip(prev["owner"], prev["repo"]))
        print(f"  already fetched          : {len(done):,} repos (skipping)")
    todo = [k for k in repos if k not in done]
    if args.max_repos:
        todo = todo[:args.max_repos]
    print(f"  to fetch this run        : {len(todo):,} repos")
    print(f"  workers                  : {args.workers}")
    config.CODE_CACHE.mkdir(parents=True, exist_ok=True)

    buffer, t0, n_done = [], time.time(), 0
    write_lock = threading.Lock()

    def flush(force=False):
        nonlocal buffer
        with write_lock:
            if buffer and (force or len(buffer) >= 200):
                df = pd.DataFrame(buffer)
                hdr = not config.FILE_CODE_INDEX.exists()
                df.to_csv(config.FILE_CODE_INDEX, mode="a", header=hdr,
                          index=False)
                buffer = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch_repo, o, r): (o, r) for o, r in todo}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="  fetching", unit="repo"):
            owner, repo = futures[fut]
            try:
                got = fut.result()
            except Exception as e:
                got = [_fail_row(owner, repo, f"error:{type(e).__name__}")]
            for g in got:
                g["server_names"] = ";".join(repos[(owner, repo)])[:300]
            with write_lock:
                buffer.extend(got)
            n_done += 1
            if n_done % 200 == 0:
                flush()
    flush(force=True)

    # ---- report -----------------------------------------------------------
    idx = pd.read_csv(config.FILE_CODE_INDEX)
    ok = idx[idx["status"] == "ok"]
    n_repos_ok = ok.groupby(["owner", "repo"]).ngroups if len(ok) else 0
    elapsed = time.time() - t0
    status_counts = idx.drop_duplicates(["owner", "repo", "status"]) \
                       .groupby("status").size().to_dict()

    summary = {
        "snapshot_label": config.SNAPSHOT_LABEL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repos_attempted": int(idx.groupby(["owner", "repo"]).ngroups),
        "repos_with_files": int(n_repos_ok),
        "files_fetched": int(len(ok)),
        "bytes_fetched": int(ok["bytes"].sum()) if len(ok) else 0,
        "status_counts": {k: int(v) for k, v in status_counts.items()},
        "elapsed_seconds": round(elapsed, 1),
    }
    save_json(summary, config.DATA_INTERIM / "code_fetch_summary.json",
              "code summary")

    lines = [
        "=" * 76, " SERVER CODE FETCH REPORT",
        f" Generated : {summary['generated_at_utc']}", "=" * 76,
        f" Repos attempted         : {summary['repos_attempted']:,}",
        f" Repos with files        : {n_repos_ok:,}",
        f" Files fetched           : {len(ok):,}",
        f" Bytes fetched           : "
        f"{(ok['bytes'].sum()/1e6) if len(ok) else 0:.1f} MB",
        f" Median files per repo   : "
        f"{ok.groupby(['owner','repo']).size().median():.0f}" if len(ok)
        else " Median files per repo   : 0",
        f" Elapsed                 : {elapsed/60:.1f} min",
        "-" * 76, " Outcome by repository:",
    ]
    for k, v in sorted(status_counts.items(), key=lambda x: -x[1]):
        lines.append(f"   {k:<22s} {v:>8,}")
    lines += ["-" * 76,
              " NOTE: not_found and forbidden repositories are registry",
              " entries whose source is private or removed. They are reported",
              " as attrition, not imputed.", "=" * 76]
    save_txt("\n".join(lines), config.DATA_INTERIM / "code_fetch_report.txt",
             "code report")

    print("\n".join(lines))
    print(f"  Index   : {config.FILE_CODE_INDEX}")
    print("  Next    : python data_processing/01_clean_and_merge.py")


if __name__ == "__main__":
    main()