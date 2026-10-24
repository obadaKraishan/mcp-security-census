"""
============================================================================
 03_detect_drift.py, Version-to-Version Drift in What a Server Advertises
============================================================================
 Project : Same Name, Different Server (MCP Security Census, IEEE BigData 2026)
 Purpose : RQ2 asks how often what a server advertises changes after
           publication. For every server with more than one published
           version, consecutive versions are compared on the three surfaces
           the registry versions: description text, package identifiers, and
           remote endpoints. Each change is classified as
             additive        - capability added, nothing withdrawn
             breaking        - something an installed client relied on is gone
             silent_semantic - identifiers unchanged while meaning or
                               destination changed (the rug-pull shape)
 Input   : data/raw/server_versions.parquet
 Output  : data/interim/drift_events.csv (+ .json + drift_report.txt)
 Usage   : python data_processing/03_detect_drift.py
           python data_processing/03_detect_drift.py --max-transitions 50
============================================================================
 Endpoint redirection (same package, new host) is the strongest signal here:
 an installed client keeps calling the server name while traffic moves to a
 different origin, with no protocol-level notification.
============================================================================
"""

import argparse
import difflib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setup import config                                    # noqa: E402
from utils.io_helpers import save_csv, save_json, save_txt  # noqa: E402

SIM_UNCHANGED = 0.98      # >= this similarity counts as unchanged
SIM_MAJOR = 0.70          # < this counts as a major rewrite


def norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def as_set(field) -> set:
    """'a|b|c' -> {'a','b','c'} (empty string yields the empty set)."""
    s = str(field or "").strip()
    return {p for p in s.split("|") if p} if s else set()


def version_key(v: str) -> tuple:
    """Numeric version ordering; tuple so pandas can sort on it."""
    parts = re.findall(r"\d+", str(v))
    nums = ([int(p) for p in parts[:4]] + [0] * 4)[:4] if parts else [0, 0, 0, 0]
    return tuple(nums)


def classify(prev: dict, cur: dict) -> list:
    """All change events between two consecutive versions."""
    ev = []

    # ---- 1. description -----------------------------------------------------
    dp, dc = norm(prev.get("description")), norm(cur.get("description"))
    if dp and dc:
        s = sim(dp, dc)
        if s < SIM_UNCHANGED:
            ev.append({"change": "description_changed",
                       "surface": "description",
                       "drift_type": "silent_semantic",
                       "similarity": round(s, 4),
                       "magnitude": "major" if s < SIM_MAJOR else "minor",
                       "detail": f"description similarity {s:.2f}"})
    elif dc and not dp:
        ev.append({"change": "description_added", "surface": "description",
                   "drift_type": "additive", "similarity": 0.0,
                   "magnitude": "minor", "detail": "description first added"})

    # ---- 2. packages --------------------------------------------------------
    pp, pc = as_set(prev.get("package_ids")), as_set(cur.get("package_ids"))
    if pp - pc:
        ev.append({"change": "package_removed", "surface": "package",
                   "drift_type": "breaking", "similarity": 1.0,
                   "magnitude": "major",
                   "detail": f"package(s) withdrawn: {sorted(pp - pc)[:3]}"})
    if pc - pp:
        ev.append({"change": "package_added", "surface": "package",
                   "drift_type": "additive", "similarity": 1.0,
                   "magnitude": "minor",
                   "detail": f"package(s) added: {sorted(pc - pp)[:3]}"})
    tp, tc = as_set(prev.get("package_types")), as_set(cur.get("package_types"))
    if pp and pc and pp == pc and tp != tc:
        ev.append({"change": "package_registry_changed", "surface": "package",
                   "drift_type": "silent_semantic", "similarity": 1.0,
                   "magnitude": "major",
                   "detail": f"same identifier, registry {sorted(tp)} -> "
                             f"{sorted(tc)}"})
    rp, rc = as_set(prev.get("runtime_hints")), as_set(cur.get("runtime_hints"))
    if rp != rc and (rp or rc):
        ev.append({"change": "runtime_hint_changed", "surface": "package",
                   "drift_type": "silent_semantic", "similarity": 1.0,
                   "magnitude": "minor",
                   "detail": f"runtime {sorted(rp)} -> {sorted(rc)}"})

    # ---- 3. remotes (the strongest rug-pull signal) --------------------------
    up, uc = as_set(prev.get("remote_urls")), as_set(cur.get("remote_urls"))
    hp, hc = as_set(prev.get("remote_hosts")), as_set(cur.get("remote_hosts"))
    if up - uc:
        ev.append({"change": "remote_removed", "surface": "remote",
                   "drift_type": "breaking", "similarity": 1.0,
                   "magnitude": "major",
                   "detail": f"endpoint(s) withdrawn: {sorted(up - uc)[:2]}"})
    if uc - up:
        ev.append({"change": "remote_added", "surface": "remote",
                   "drift_type": "additive", "similarity": 1.0,
                   "magnitude": "minor",
                   "detail": f"endpoint(s) added: {sorted(uc - up)[:2]}"})
    # same count, different host: traffic redirected to a new origin
    if hp and hc and hp != hc and len(hp) == len(hc):
        ev.append({"change": "remote_host_redirected", "surface": "remote",
                   "drift_type": "silent_semantic", "similarity": 1.0,
                   "magnitude": "major",
                   "detail": f"host {sorted(hp)} -> {sorted(hc)}"})
    yp, yc = as_set(prev.get("remote_types")), as_set(cur.get("remote_types"))
    if yp and yc and yp != yc:
        ev.append({"change": "transport_changed", "surface": "remote",
                   "drift_type": "silent_semantic", "similarity": 1.0,
                   "magnitude": "minor",
                   "detail": f"transport {sorted(yp)} -> {sorted(yc)}"})

    # ---- 4. tool definitions (only when the record embeds them) -------------
    try:
        pt = {t["name"]: t.get("description", "")
              for t in json.loads(prev.get("tools_json") or "[]")}
        ct = {t["name"]: t.get("description", "")
              for t in json.loads(cur.get("tools_json") or "[]")}
        for n in set(pt) & set(ct):
            s = sim(norm(pt[n]), norm(ct[n]))
            if s < SIM_UNCHANGED:
                ev.append({"change": "tool_description_changed",
                           "surface": "tool", "drift_type": "silent_semantic",
                           "similarity": round(s, 4),
                           "magnitude": "major" if s < SIM_MAJOR else "minor",
                           "detail": f"tool '{n}' description similarity {s:.2f}"})
        for n in set(pt) - set(ct):
            ev.append({"change": "tool_removed", "surface": "tool",
                       "drift_type": "breaking", "similarity": 1.0,
                       "magnitude": "major", "detail": f"tool '{n}' removed"})
        for n in set(ct) - set(pt):
            ev.append({"change": "tool_added", "surface": "tool",
                       "drift_type": "additive", "similarity": 1.0,
                       "magnitude": "minor", "detail": f"tool '{n}' added"})
    except (json.JSONDecodeError, TypeError, KeyError):
        pass

    return ev


def main():
    ap = argparse.ArgumentParser(description="Detect version-to-version drift")
    ap.add_argument("--max-transitions", type=int, default=None,
                    help="cap transitions per server so bulk republishers "
                         "cannot dominate the counts")
    args = ap.parse_args()

    print("=" * 76)
    print(" STEP 3/4 (PROCESSING) — VERSION DRIFT")
    print("=" * 76)
    if not config.FILE_VERSIONS.exists():
        sys.exit("[FAIL] server_versions.parquet missing — run collection 02")
    ver = pd.read_parquet(config.FILE_VERSIONS)
    print(f"  version records          : {len(ver):,}")
    needed = {"package_ids", "remote_urls", "remote_hosts"}
    if not needed.issubset(ver.columns):
        sys.exit("[FAIL] version table lacks package/remote fields — re-run "
                 "data_collection/02_fetch_version_history.py "
                 "(delete data/.api_cache/ver_*.json first is NOT needed; "
                 "the fetcher re-parses cached payloads)")

    counts = ver.groupby("name").size()
    multi = counts[counts > 1].index
    print(f"  servers with >1 version  : {len(multi):,}   <- RQ2 analysis set")
    if args.max_transitions:
        print(f"  transition cap per server: {args.max_transitions}")

    ver = ver[ver["name"].isin(multi)].copy()
    ver["published_dt"] = pd.to_datetime(ver["published_at"], utc=True,
                                         errors="coerce", format="mixed")
    ver["vkey"] = ver["version"].map(version_key)

    rows, n_transitions = [], 0
    for name, grp in tqdm(ver.groupby("name"), total=len(multi),
                          desc="  comparing", unit="srv"):
        recs = (grp.sort_values(["published_dt", "vkey"], na_position="last")
                   .to_dict("records"))
        pairs = list(zip(recs, recs[1:]))
        if args.max_transitions:
            pairs = pairs[:args.max_transitions]
        n_transitions += len(pairs)
        for prev, cur in pairs:
            events = classify(prev, cur)
            if not events:
                continue
            days = None
            if pd.notna(prev.get("published_dt")) and \
               pd.notna(cur.get("published_dt")):
                days = (cur["published_dt"] - prev["published_dt"]).days
            for e in events:
                rows.append({"server_name": name,
                             "from_version": prev.get("version"),
                             "to_version": cur.get("version"),
                             "from_published": str(prev.get("published_at")),
                             "to_published": str(cur.get("published_at")),
                             "days_between": days, **e})

    ddf = pd.DataFrame(rows)
    save_csv(ddf, config.FILE_DRIFT, "drift events")

    # ---- report -----------------------------------------------------------
    def srv_n(mask=None):
        d = ddf if mask is None else ddf[mask]
        return d["server_name"].nunique() if len(d) else 0

    n_any = srv_n()
    n_silent = srv_n(ddf["drift_type"] == "silent_semantic") if len(ddf) else 0
    n_break = srv_n(ddf["drift_type"] == "breaking") if len(ddf) else 0
    n_redirect = srv_n(ddf["change"] == "remote_host_redirected") if len(ddf) else 0
    n_pkg_pull = srv_n(ddf["change"] == "package_removed") if len(ddf) else 0
    by_type = ddf["drift_type"].value_counts().to_dict() if len(ddf) else {}
    by_change = ddf["change"].value_counts().to_dict() if len(ddf) else {}
    by_surface = ddf["surface"].value_counts().to_dict() if len(ddf) else {}
    M = max(len(multi), 1)

    lines = [
        "=" * 76, " VERSION DRIFT REPORT",
        f" Generated : {datetime.now(timezone.utc).isoformat()}",
        f" Snapshot  : {config.SNAPSHOT_LABEL}", "=" * 76,
        f" Multi-version servers      : {len(multi):,}",
        f" Version transitions compared: {n_transitions:,}",
        f" Drift events detected       : {len(ddf):,}",
        "-" * 76,
        f" Servers with any drift      : {n_any:,} ({n_any/M:.1%})",
        f"   silent-semantic drift     : {n_silent:,} ({n_silent/M:.1%})",
        f"   breaking drift            : {n_break:,} ({n_break/M:.1%})",
        "-" * 76, " Highest-risk signals:",
        f"   endpoint host redirected  : {n_redirect:,} ({n_redirect/M:.2%})",
        f"   package withdrawn         : {n_pkg_pull:,} ({n_pkg_pull/M:.2%})",
        "-" * 76, " Events by drift type:",
    ]
    for k, v in by_type.items():
        lines.append(f"   {k:<28s} {v:>8,}")
    lines.append(" Events by surface:")
    for k, v in by_surface.items():
        lines.append(f"   {k:<28s} {v:>8,}")
    lines.append(" Events by change kind:")
    for k, v in by_change.items():
        lines.append(f"   {k:<28s} {v:>8,}")
    if len(ddf) and ddf["days_between"].notna().any():
        lines += ["-" * 76,
                  f" Median days between changed versions: "
                  f"{ddf['days_between'].median():.0f}"]
    lines += ["-" * 76,
              " NOTE: the registry does not version tool definitions, so a",
              " client cannot detect a tool-behaviour change from the registry",
              " alone. Drift is measured on description, package, and remote",
              " endpoint fields; endpoint redirection is the closest",
              " observable analogue of a rug pull.", "=" * 76]

    save_json({"multi_version_servers": int(len(multi)),
               "transitions_compared": int(n_transitions),
               "drift_events": int(len(ddf)),
               "servers_with_drift": int(n_any),
               "servers_with_silent_drift": int(n_silent),
               "servers_with_breaking_drift": int(n_break),
               "servers_with_host_redirect": int(n_redirect),
               "by_drift_type": {k: int(v) for k, v in by_type.items()},
               "by_change": {k: int(v) for k, v in by_change.items()},
               "generated_at_utc": datetime.now(timezone.utc).isoformat()},
              config.DATA_INTERIM / "drift_summary.json", "drift summary")
    save_txt("\n".join(lines), config.DATA_INTERIM / "drift_report.txt",
             "drift report")
    print("\n".join(lines))
    print("  Next: python data_processing/04_feature_engineering.py")


if __name__ == "__main__":
    main()
