"""
============================================================================
 config.py, Single Source of Truth for the MCP Security Census
============================================================================
 Project : Same Name, Different Server, A Security and Maintainability Census
           of the Model Context Protocol Server Ecosystem
 Purpose : Central constants imported by EVERY script: paths, registry
           endpoints, the THREAT CATALOGUE, severity mapping, sampling
           rules, and statistics settings. Nothing is hard-coded twice.
 Usage   : from setup.config import *      (from project root)
           python setup/config.py          (standalone: prints config)
============================================================================
"""

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. PROJECT PATHS (derived from repo root, portable on any machine)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_INTERIM = DATA_DIR / "interim"
DATA_PROCESSED = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / ".api_cache"          # registry + GitHub responses
CODE_CACHE = DATA_DIR / ".code_cache"        # fetched server source files

ANALYSIS_DIR = PROJECT_ROOT / "analysis"
LABELING_DIR = PROJECT_ROOT / "labeling"
CODEBOOK_DIR = PROJECT_ROOT / "codebook"
PAPER_DIR = PROJECT_ROOT / "paper"
PAPER_TABLES = PAPER_DIR / "tables"
PAPER_FIGURES = PAPER_DIR / "figures"

ALL_DIRS = [DATA_RAW, DATA_INTERIM, DATA_PROCESSED, CACHE_DIR, CODE_CACHE,
            ANALYSIS_DIR, LABELING_DIR, CODEBOOK_DIR, PAPER_TABLES, PAPER_FIGURES]

# ---------------------------------------------------------------------------
# 2. DATA SOURCES
# ---------------------------------------------------------------------------
# Official MCP Registry (public, no auth). Cursor pagination; the Aug 2026
# snapshot holds 21,643 unique servers and 72,606 version records.
REGISTRY_BASE = "https://registry.modelcontextprotocol.io"
REGISTRY_SERVERS = f"{REGISTRY_BASE}/v0/servers"
REGISTRY_PAGE_SIZE = 100          # max supported by the endpoint
REGISTRY_TIMEOUT_S = 30
REGISTRY_MAX_RETRIES = 4

# GitHub (popularity/maintenance signals + source code for repo-linked servers)
GITHUB_API_BASE = "https://api.github.com"
GITHUB_PER_PAGE = 100
GITHUB_TIMEOUT_S = 30
GITHUB_MAX_RETRIES = 5

# Code-fetch limits (keeps the crawl bounded and reproducible)
MAX_FILES_PER_REPO = 40           # scan the most relevant files, not whole trees
MAX_FILE_BYTES = 200_000          # skip vendored bundles / lockfiles
CODE_EXTENSIONS = {".py", ".js", ".ts", ".mjs", ".cjs", ".tsx", ".jsx",
                   ".json", ".yaml", ".yml", ".toml"}
MANIFEST_FILES = {"package.json", "pyproject.toml", "requirements.txt",
                  "Dockerfile", "smithery.yaml", "mcp.json", "server.json"}

# ---------------------------------------------------------------------------
# 3. SNAPSHOT IDENTITY (the registry changes weekly, so the snapshot label travels with every output)
# ---------------------------------------------------------------------------
SNAPSHOT_LABEL = "2026-08"        # bump when you re-harvest
SCHEMA_NOTE = "server.schema.json 2025-12-11 observed in Aug 2026 pull"

# ---------------------------------------------------------------------------
# 4. THREAT CATALOGUE 
# ---------------------------------------------------------------------------
# Each class: human-readable name, where it is detected, severity band, and
# the patterns that flag it. Detection is PATTERN-BASED: hits are risk
# INDICATORS, not confirmed vulnerabilities, and every report says so.
#
# surface: "tool"     -> tool names/descriptions/schemas from server metadata
#          "code"     -> handler source files
#          "manifest" -> package/config manifests
#          "meta"     -> registry record fields (transport, remotes, packages)
# severity: informational | low | medium | high  (mapped to a score below)

THREAT_CLASSES = {
    "TP_tool_poisoning": {
        "name": "Tool poisoning (hidden instructions in tool metadata)",
        "surface": "tool",
        "severity": "high",
        "patterns": [
            # override of prior instructions, excluding defensive warnings
            r"""(?im)^(?![^\n]*(untrusted|never treat|do not follow|not an instruction|ignore any instructions in))[^\n]*\b(ignore|disregard|forget)\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts?)""",
            # instruction to conceal an action from the user
            r"""(?im)^(?![^\n]*(untrusted|never treat|warning|security note))[^\n]*\b(do\s+not|don't|never)\s+(tell|inform|mention\s+to|reveal\s+to)\s+(the\s+)?(user|human|operator)""",
            # imperative to read sensitive local files before acting
            r"""(?i)\b(first|before)\b[^.\n]{0,60}\b(read|open|cat|load)\b[^.\n]{0,40}(\.ssh|id_rsa|\.env|credentials)""",
            # emphasis tag wrapping a directive aimed at the model
            r"""(?is)<\s*(important|system|secret)\s*>[^<]{0,200}\b(must|always|never|do not)\b""",
        ],
    },
    "AA_ambient_authority": {
        "name": "Ambient authority (over-broad or unscoped permissions)",
        "surface": "code",
        "severity": "medium",
        "patterns": [
            # wildcard scope/permission/origin assignment; skip TS type unions
            r"""(?im)^(?![^\n]*[|])[^\n]*["']?\b(scopes?|permissions?|allowed[_-]?(directories|paths|origins|tools))\b["']?\s*[:=]\s*[\[\(]?\s*["']\*["']""",
            # filesystem root granted as an allowed directory
            r"""(?im)^[^\n]*["']?\b(allowed[_-]?(directories|paths)|ALLOWED_PATHS)\b["']?\s*[:=]\s*[\[\(]?\s*["'][/\\]["']""",
            # admin/full value on a permission key; skip DB queries and CLI help
            r"""(?im)^(?![^\n]*(--\w|\bwhere\b|\.count\(|\.find|[|]))[^\n]*["']?\b(scopes?|permissions?|privileges?)\b["']?\s*[:=]\s*[\[\(]?\s*["'](admin|root|full[_-]?access|all|\*)["']""",
            # dangerous OAuth scope combination inside one quoted value
            r"""(?i)["'](repo|admin:org|write:org)(\s*,\s*[\w:]+)*\s*,\s*(delete_repo|admin:org)[\w:,\s]*["']""",
        ],
    },
    "TPT_token_passthrough": {
        "name": "Token passthrough / confused deputy",
        "surface": "code",
        "severity": "high",
        "patterns": [
            r"(?i)headers?\s*\[\s*[\"\']authorization[\"\']\s*\]\s*=\s*[^=\n]*\b(req|request|params|args|input|ctx|context)\b",
            r"(?i)Bearer\s+[\$\{]*\s*(req|request|params|args|input|ctx|context|body|query)\s*[.\[]\s*(\w+)\s*(?![\(\w])",
            r"(?i)[\"\']?Authorization[\"\']?\s*:\s*[`\"\'][^`\"\'\n]*\$\{\s*(req|request|params|args|input|body)\s*[.\[]\s*\w+(?!\s*\()",
            r"(?i)\b(forward|passthrough|relay)[_-]?(auth|token|credential|header)s?\b\s*[:=(]",
            r"(?i)\bpass(es|ing)?[_-]?through[_-]?(token|auth)\b",
        ],
    },
    "CI_command_injection": {
        "name": "Command / code injection in handlers",
        "surface": "code",
        "severity": "high",
        "patterns": [
            # shell=True as real code; exclude comments, strings and backticks
            r"""(?m)^[^#\n"'`]*\bshell\s*=\s*True\b""",
            r"""(?i)\b(os\.(system|popen))\s*\([^)\n]*(\+|%\s|f["']|\.format\(|\$\{)""",
            r"""(?i)\bsubprocess\.(call|run|Popen)\s*\(\s*(f["']|["'][^"'\n]*["']\s*(\+|%)|\w+\s*\+\s*["'])""",
            r"""(?i)\bchild_process\.(exec|execSync)\s*\(\s*[`"'][^`"'\n]*\$\{""",
            r"""(?<![\w.])(eval|exec)\s*\(\s*(req|request|params|args|input|tool|user)\b""",
        ],
    },
    "UE_unauthenticated_exposure": {
        "name": "Unauthenticated network exposure",
        "surface": "code",
        "severity": "high",
        "patterns": [
            # binding to all interfaces in real code (not a comment)
            r"(?m)^[^#\n]*\.listen\s*\(\s*[^,)\n]+,\s*[\"\']0\.0\.0\.0[\"\']",
            r"(?m)^[^#\n]*\bhost\s*=\s*[\"\']0\.0\.0\.0[\"\']",
            # authentication explicitly disabled as a setting
            r"(?im)^(?![\s]*[#*/])[^#\n]*\b(auth|authentication|require[_-]?auth|auth[_-]?required)\s*[:=]\s*(False|false|None|null|0)\b",
            # permissive CORS assigned in code
            r"(?m)^[^#\n]*[\"\']?Access-Control-Allow-Origin[\"\']?\s*[:=]\s*[\"\']\*[\"\']",
        ],
    },
    "SEC_insecure_secrets": {
        "name": "Insecure secret handling",
        "surface": "code",
        "severity": "medium",
        "patterns": [
            # literal secret value; reject ENV names, placeholders, test fixtures
            r"""(?i)\b(api[_-]?key|secret|auth[_-]?token|password|passwd)\s*[:=]\s*["'](?![A-Z0-9]+(_[A-Z0-9]+)+["'])(?![^"']*(example|placeholder|your[_-]|xxx|changeme|test-|test_|mock|fake|dummy|sample|canary|redact))[A-Za-z0-9_\-]{16,}["']""",
            # live credential formats, excluding vendor documentation examples
            r"""(?!.*EXAMPLE)\b(sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|AKIA(?!IOSFODNN7EXAMPLE)[0-9A-Z]{16})\b""",
            # logging a secret IDENTIFIER (bare variable, not text in a string)
            r"""(?i)(console\.log|print)\s*\(\s*[^"'\n)]*\b(api[_-]?key|apiKey|access[_-]?token|secret_key|password)\b\s*[,)]""",
        ],
    },
    "DEP_unpinned_dependency": {
        "name": "Unpinned or unverified dependency",
        "surface": "manifest",
        "severity": "low",
        "patterns": [
            # dependency pinned to a floating tag; exclude non-package keys
            r"""(?i)["'](?!(tag|node|npm|engines?|version|type|main|license|name)["'])[@\w][\w@/.\-]*["']\s*:\s*["'](\*|latest|\^?[0-9]+\.x)["']""",
            r"""(?i)\bcurl\s+[^|\n]*\|\s*(ba)?sh\b""",
            # bare requirement line, never a shell keyword
            r"""(?im)^(?!(fi|esac|done|else|then|do|exit|set|cd|rm|cp|mv|EOF|END)\s*$)[a-z][a-z0-9_.\-]{2,}\s*$""",
        ],
    },
    "SSRF_unvalidated_fetch": {
        "name": "Unvalidated outbound fetch (SSRF surface)",
        "surface": "code",
        "severity": "medium",
        "patterns": [
            # fetch target from request data; not a property lookup (this.requests)
            r"""(?i)(?<![.\w])(requests\.(get|post|put)|axios\.(get|post)|fetch)\s*\(\s*(req|request|params|args|input|body|query)\s*[.\[]""",
            r"""(?i)(?<![.\w])(requests\.(get|post)|urlopen|fetch)\s*\(\s*(user_?url|target_?url|input_?url)\b""",
            # TLS verification disabled in real code, not comments
            r"""(?m)^[^#\n]*\b(verify\s*=\s*False|rejectUnauthorized\s*:\s*false)\b""",
        ],
    },
}

# Severity -> numeric weight for the per-server risk score
SEVERITY_WEIGHT = {"informational": 0, "low": 1, "medium": 3, "high": 5}

# Compiled once at import (used by data_processing/02_scan_security.py)
COMPILED_THREATS = {
    cls: [re.compile(p) for p in spec["patterns"]]
    for cls, spec in THREAT_CLASSES.items()
}

# Drift classification for RQ2 (tool-definition changes across versions)
DRIFT_TYPES = {
    "additive": "new tool or new optional field added",
    "breaking": "tool removed or required field changed",
    "silent_semantic": "description/behavior text changed while name and "
                       "schema stayed the same (rug-pull-like)",
}

# ---------------------------------------------------------------------------
# 5. SAMPLING (RQ4 ground truth)
# ---------------------------------------------------------------------------
LABEL_SAMPLE_SIZE = 300           # stratified manual-labeling target
LABEL_STRATA_MIN = 20             # minimum servers per stratum where available
LABEL_POSITIVE_SHARE = 0.5        # balance flagged vs unflagged for recall est.

# ---------------------------------------------------------------------------
# 6. STATISTICS
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
ALPHA = 0.05
N_BOOTSTRAP = 5000
CI_LEVEL = 0.95
P_ADJUST_METHOD = "fdr_bh"        # Benjamini-Hochberg

# Robustness settings (RQ2 association and redirection, RQ3 popularity)
DRIFT_TRANSITION_CAP = 20         # cap per server for the bulk-republisher check
N_SUBSAMPLE_DRAWS = 100           # repeated one-per-publisher draws

CLIFFS_DELTA_BANDS = [(0.147, "negligible"), (0.33, "small"),
                      (0.474, "medium"), (float("inf"), "large")]
CRAMERS_V_BANDS = [(0.1, "negligible"), (0.3, "small"),
                   (0.5, "medium"), (float("inf"), "large")]

# ---------------------------------------------------------------------------
# 7. RESEARCH QUESTION REGISTRY
# ---------------------------------------------------------------------------
RESEARCH_QUESTIONS = {
    "rq1": {"dir": "rq1_threat_prevalence",
            "title": "RQ1 — Threat-Class Prevalence Across the Server Population",
            "question": "How prevalent is each MCP threat class across the "
                        "public server population?"},
    "rq2": {"dir": "rq2_tool_drift",
            "title": "RQ2 — Silent Tool-Definition Drift Across Versions",
            "question": "How often do tool definitions change silently across "
                        "published versions?"},
    "rq3": {"dir": "rq3_popularity_security",
            "title": "RQ3 — Popularity and Security Posture",
            "question": "Does popularity predict a server's security posture?"},
    "rq4": {"dir": "rq4_scanner_accuracy",
            "title": "RQ4 — Scanner Accuracy Against Manual Ground Truth",
            "question": "How accurately does the automated scanner flag risky "
                        "servers?"},
    "rq5": {"dir": "rq5_ambient_authority",
            "title": "RQ5 — Ambient Authority and Token Passthrough",
            "question": "What share of servers expose ambient authority or "
                        "pass tokens through?"},
}

def rq_output_dir(k: str) -> Path:
    return ANALYSIS_DIR / RESEARCH_QUESTIONS[k]["dir"] / "outputs"

def rq_figure_dir(k: str) -> Path:
    return ANALYSIS_DIR / RESEARCH_QUESTIONS[k]["dir"] / "figures"

# ---------------------------------------------------------------------------
# 8. PIPELINE FILE CONTRACT
# ---------------------------------------------------------------------------
FILE_REGISTRY_RAW    = DATA_RAW / "registry_servers.parquet"       # collect/01
FILE_REGISTRY_MANIFEST = DATA_RAW / "registry_manifest.json"       # collect/01
FILE_VERSIONS        = DATA_RAW / "server_versions.parquet"        # collect/02
FILE_CODE_INDEX      = DATA_INTERIM / "code_index.csv"             # collect/03
FILE_SERVERS_MERGED  = DATA_INTERIM / "servers_merged.csv"         # process/01
FILE_FINDINGS        = DATA_INTERIM / "security_findings.csv"      # process/02
FILE_DRIFT           = DATA_INTERIM / "drift_events.csv"           # process/03
FILE_MASTER          = DATA_PROCESSED / "master.csv"               # process/04
FILE_MASTER_JSON     = DATA_PROCESSED / "master_summary.json"      # process/04
FILE_LABEL_SHEET     = LABELING_DIR / "labeling_sheet.csv"         # labeling/01
FILE_LABEL_DONE      = LABELING_DIR / "labeling_sheet_completed.csv"  # manual
FILE_SCANNER_SCORES  = LABELING_DIR / "scanner_accuracy.csv"       # labeling/02

# ---------------------------------------------------------------------------
# Standalone: print the resolved configuration
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 76)
    print(" MCP SECURITY CENSUS — Resolved Configuration")
    print("=" * 76)
    print(f"  PROJECT_ROOT     : {PROJECT_ROOT}")
    print(f"  Registry         : {REGISTRY_SERVERS}")
    print(f"  Snapshot label   : {SNAPSHOT_LABEL}")
    print(f"  Page size        : {REGISTRY_PAGE_SIZE}")
    print(f"  Code fetch caps  : {MAX_FILES_PER_REPO} files/repo, "
          f"{MAX_FILE_BYTES:,} bytes/file")
    print(f"  Random seed      : {RANDOM_SEED}   Alpha: {ALPHA}   "
          f"CI: {CI_LEVEL}   Bootstrap: {N_BOOTSTRAP:,}")
    print("-" * 76)
    print(f"  THREAT CATALOGUE — {len(THREAT_CLASSES)} classes, "
          f"{sum(len(s['patterns']) for s in THREAT_CLASSES.values())} patterns")
    for cls, spec in THREAT_CLASSES.items():
        print(f"    {cls:<28s} {spec['severity']:<8s} "
              f"{spec['surface']:<9s} {len(spec['patterns'])} patterns")
    print(f"  Severity weights : {SEVERITY_WEIGHT}")
    print(f"  Drift types      : {', '.join(DRIFT_TYPES)}")
    print("-" * 76)
    print("  Research questions:")
    for k, rq in RESEARCH_QUESTIONS.items():
        print(f"    {k.upper():4s} -> analysis/{rq['dir']}/")
    print("=" * 76)
    print("  Config OK — import with:  from setup.config import *")
