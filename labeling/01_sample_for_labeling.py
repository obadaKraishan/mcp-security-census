"""
============================================================================
 01_sample_for_labeling.py, Build the Manual Ground-Truth Sheet (RQ4)
============================================================================
 Project : Same Name, Different Server (MCP Security Census, IEEE BigData 2026)
 Purpose : A scanner without measured accuracy is not publishable. This
           script draws a stratified sample of findings and writes a
           labelling sheet the analyst fills in by hand. Two strata matter:
             FLAGGED   - findings the scanner produced (measures PRECISION)
             UNFLAGGED - random scanned files with no finding (measures
                         RECALL: how many real issues the scanner missed)
           Without the second stratum recall cannot be estimated at all.
 Input   : data/interim/security_findings.csv, code_index.csv, master.csv
 Output  : labeling/labeling_sheet.csv   (fill in the 'label' column)
           labeling/labeling_guide.txt   (decision rules for each class)
 Usage   : python labeling/01_sample_for_labeling.py
           python labeling/01_sample_for_labeling.py --per-class 40 --unflagged 100
============================================================================
 The sheet is deliberately small and self-contained: each row carries the
 evidence line, the file, and a direct link, so a decision takes seconds.
============================================================================
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setup import config                                    # noqa: E402
from utils.io_helpers import save_csv, save_txt             # noqa: E402

# Decision rules the analyst applies. Deliberately concrete: the same rule
# must be applicable months later and by a second rater.
RULES = {
    "TP_tool_poisoning": (
        "TRUE if the tool name/description contains text aimed at steering the "
        "model rather than describing the tool: overriding earlier "
        "instructions, telling the model to conceal an action, or directing it "
        "to read sensitive files. FALSE if it is ordinary documentation, or a "
        "defensive warning that page content is untrusted."),
    "AA_ambient_authority": (
        "TRUE if a permission, scope, allowed-directory, or origin setting "
        "actually grants unrestricted access (\"*\", \"/\", admin) in code the "
        "server runs. FALSE for prose, help text, type declarations, database "
        "queries, or a variable that merely happens to be called scope/root."),
    "TPT_token_passthrough": (
        "TRUE if a credential taken from the caller (request, params, input, "
        "ctx) is placed into an outbound Authorization header, so the server "
        "acts on a token it did not mint. FALSE if the token comes from the "
        "server's own environment or config."),
    "CI_command_injection": (
        "TRUE if a shell command is executed with shell=True or built by "
        "concatenating or interpolating values into a command string. FALSE "
        "for list-form argv calls, and for mentions inside comments, "
        "docstrings, or security-check scripts."),
    "UE_unauthenticated_exposure": (
        "TRUE if the server binds to 0.0.0.0, disables authentication, or sets "
        "a wildcard CORS origin in code that runs. FALSE if it appears in a "
        "comment, or is clearly a local-development-only branch."),
    "SEC_insecure_secrets": (
        "TRUE if a real credential value is embedded in source, or a secret "
        "value is written to logs. FALSE for environment-variable names, "
        "placeholders, documented vendor examples, and redaction tests."),
    "DEP_unpinned_dependency": (
        "TRUE if a dependency is declared with no version constraint or a "
        "floating tag (*, latest, N.x), or a remote script is piped to a "
        "shell. FALSE for engine/runtime fields and non-dependency keys."),
    "SSRF_unvalidated_fetch": (
        "TRUE if an outbound request targets a URL taken from caller input, or "
        "TLS verification is disabled in code that runs. FALSE for property "
        "lookups that merely resemble a fetch, and for comments."),
}

HEADER_NOTE = (
    "Fill in the 'label' column with 1 (true finding) or 0 (false positive). "
    "Use 'notes' for anything ambiguous. For UNFLAGGED rows, label 1 only if "
    "the file does contain an issue of that class that the scanner missed."
)


def build_link(row) -> str:
    owner, repo, path = row.get("owner"), row.get("repo"), row.get("file", "")
    if not owner or not repo:
        return ""
    return f"https://github.com/{owner}/{repo}/blob/HEAD/{path}"


def main():
    ap = argparse.ArgumentParser(description="Build the RQ4 labelling sheet")
    ap.add_argument("--per-class", type=int, default=None,
                    help="findings sampled per threat class "
                         "(default: config.LABEL_SAMPLE_SIZE split evenly)")
    ap.add_argument("--unflagged", type=int, default=120,
                    help="unflagged files sampled for recall estimation")
    args = ap.parse_args()

    print("=" * 76)
    print(" LABELLING SAMPLE — ground truth for RQ4")
    print("=" * 76)
    if not config.FILE_FINDINGS.exists():
        sys.exit("[FAIL] security_findings.csv missing — run the scanner")
    fd = pd.read_csv(config.FILE_FINDINGS, low_memory=False)
    print(f"  findings available      : {len(fd):,}")

    present = [c for c in config.THREAT_CLASSES
               if (fd["threat_class"] == c).any()]
    empty = [c for c in config.THREAT_CLASSES if c not in present]
    print(f"  classes with findings   : {len(present)}")
    if empty:
        print(f"  classes with none       : {', '.join(empty)}")
        print("     (zero-prevalence classes cannot enter precision/recall;")
        print("      report them with a Wilson upper bound instead)")

    per_class = args.per_class or max(
        20, config.LABEL_SAMPLE_SIZE // max(len(present), 1))
    rng = np.random.default_rng(config.RANDOM_SEED)

    # ---- stratum 1: FLAGGED findings (precision) --------------------------
    rows = []
    for cls in present:
        sub = fd[fd["threat_class"] == cls]
        # sample distinct servers first so one noisy repo cannot dominate
        servers = sub["server_name"].drop_duplicates()
        pick_srv = servers.sample(
            n=min(per_class, len(servers)), random_state=config.RANDOM_SEED)
        take = (sub[sub["server_name"].isin(pick_srv)]
                .groupby("server_name", group_keys=False)
                .head(1))
        if len(take) > per_class:
            take = take.sample(n=per_class, random_state=config.RANDOM_SEED)
        for _, r in take.iterrows():
            rows.append({
                "stratum": "flagged",
                "threat_class": cls,
                "severity": config.THREAT_CLASSES[cls]["severity"],
                "server_name": r.get("server_name", ""),
                "owner": r.get("owner", ""),
                "repo": r.get("repo", ""),
                "file": r.get("file", ""),
                "line": r.get("line", ""),
                "evidence": str(r.get("evidence", ""))[:200],
                "link": build_link(r),
                "label": "",
                "notes": "",
            })
        print(f"    {cls:<30s} sampled {len(take):>3} of "
              f"{sub['server_name'].nunique():,} servers")

    # ---- stratum 2: UNFLAGGED files (recall) ------------------------------
    n_unflagged = 0
    if config.FILE_CODE_INDEX.exists():
        idx = pd.read_csv(config.FILE_CODE_INDEX, low_memory=False)
        idx = idx[idx["status"] == "ok"]
        flagged_pairs = set(zip(fd["owner"].astype(str),
                                fd["repo"].astype(str),
                                fd["file"].astype(str)))
        cand = idx[~idx.apply(
            lambda r: (str(r["owner"]), str(r["repo"]), str(r["path"]))
            in flagged_pairs, axis=1)]
        # bias toward files likely to contain server logic
        likely = cand[cand["path"].str.contains(
            r"server|index|main|tool|handler|app", case=False, na=False)]
        pool = likely if len(likely) >= args.unflagged else cand
        take = pool.sample(n=min(args.unflagged, len(pool)),
                           random_state=config.RANDOM_SEED)
        for _, r in take.iterrows():
            rows.append({
                "stratum": "unflagged",
                "threat_class": "ANY",
                "severity": "-",
                "server_name": str(r.get("server_names", ""))[:80],
                "owner": r.get("owner", ""),
                "repo": r.get("repo", ""),
                "file": r.get("path", ""),
                "line": "",
                "evidence": "(no finding produced — inspect the file)",
                "link": (f"https://github.com/{r['owner']}/{r['repo']}"
                         f"/blob/HEAD/{r['path']}"),
                "label": "",
                "notes": "",
            })
        n_unflagged = len(take)
        print(f"    {'UNFLAGGED (recall stratum)':<30s} sampled "
              f"{n_unflagged:>3} files")

    sheet = pd.DataFrame(rows)
    sheet = sheet.sample(frac=1.0, random_state=config.RANDOM_SEED)  # shuffle
    sheet.insert(0, "row_id", range(1, len(sheet) + 1))
    save_csv(sheet, config.FILE_LABEL_SHEET, "labelling sheet")

    # ---- guide -------------------------------------------------------------
    g = ["=" * 78, " LABELLING GUIDE — RQ4 ground truth",
         f" Snapshot : {config.SNAPSHOT_LABEL}",
         f" Generated: {datetime.now(timezone.utc).isoformat()}", "=" * 78, "",
         " HOW TO USE", "-" * 78,
         " 1. Open labeling/labeling_sheet.csv in a spreadsheet.",
         " 2. For each row read 'evidence'; open 'link' when unsure.",
         " 3. Put 1 (true finding) or 0 (false positive) in 'label'.",
         " 4. Save as labeling/labeling_sheet_completed.csv.",
         " 5. Run labeling/02_score_scanner.py.", "",
         " The sheet is shuffled so classes are interleaved; this reduces the",
         " tendency to apply a different standard once a pattern is noticed.",
         "", " STRATA", "-" * 78,
         " flagged   : the scanner produced this finding. Label 1 if it is a",
         "             real instance of the class. These rows give PRECISION.",
         " unflagged : the scanner produced nothing for this file. Label 1 only",
         "             if the file does contain an issue in any catalogue class",
         "             (say which in 'notes'). These rows give RECALL.",
         "", " DECISION RULES BY CLASS", "-" * 78]
    for cls, rule in RULES.items():
        if cls not in present and cls != "ANY":
            continue
        g += [f" {cls}", f"   {rule}", ""]
    g += [" GENERAL PRINCIPLES", "-" * 78,
          " - Judge the code as written, not the developer's intent.",
          " - Comments, docstrings, and prose are never true findings.",
          " - A pattern in a security-checking script is not a finding.",
          " - When genuinely undecidable, label 0 and explain in 'notes';",
          "   this keeps precision conservative rather than flattering.",
          " - Do not revise earlier labels after forming a theory; note the",
          "   doubt instead, so the measurement stays honest.", "=" * 78]
    save_txt("\n".join(g), config.LABELING_DIR / "labeling_guide.txt", "guide")

    print("-" * 76)
    print(f"  sheet rows              : {len(sheet):,} "
          f"({len(sheet) - n_unflagged:,} flagged + {n_unflagged:,} unflagged)")
    print(f"  estimated time          : ~{len(sheet) * 20 / 3600:.1f} h "
          f"at 20 s per row")
    print(f"  sheet                   : {config.FILE_LABEL_SHEET}")
    print(f"  guide                   : {config.LABELING_DIR / 'labeling_guide.txt'}")
    print()
    print("  NEXT: fill in the 'label' column, save as")
    print(f"        {config.FILE_LABEL_DONE.name}, then run")
    print("        python labeling/02_score_scanner.py")


if __name__ == "__main__":
    main()
