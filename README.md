# Same Name, Different Server
### A Security Census of Silent Drift in the Model Context Protocol Ecosystem

Replication package for the paper accepted at the **IEEE International
Conference on Big Data 2026** (regular paper). It contains the full pipeline
that harvests the public MCP registry, fetches server source from GitHub,
applies an eight-class pattern-based threat scanner, measures version-to-
version drift, scores the scanner against hand-labeled ground truth, and
produces every number, table, and figure in the paper.

**Snapshot:** August 2026. 21,643 servers, 72,606 version records, 14,353
servers with fetched source (the scan population), 8,900 multi-version
servers (the drift population), 414 hand-labeled rows (the validation
sample).

## What is in this repository

| Folder | Contents |
|---|---|
| `setup/` | `config.py`, the single source of truth: threat catalogue (31 patterns), severity weights, drift definitions, fetch limits, sampling constants, statistical settings. `setup_environment.py` checks the environment before any collection. |
| `data_collection/` | Registry harvest (cursor pagination), version-history fetch, and the GitHub code fetch (cached, resumable, 40 files and 200 KB per file per repository). |
| `data_processing/` | Clean and merge, the scanner, drift detection, feature engineering into `master.csv`. |
| `data_validation/` | Quality gate: schema, NaN discipline, value ranges, consistency with the findings file, sampled evidence per class. Exits non-zero on failure. |
| `labeling/` | Ground-truth protocol (`labeling_guide.txt`), the sampling script, the completed label sheet, and the scoring script that produces per-class precision, the miss rate, and precision-corrected prevalence. |
| `codebook/` | Generated variable dictionary for all 68 columns of `master.csv`. |
| `analysis/rq*/` | One self-contained folder per analysis. Each writes `rqN_results.csv`, `rqN_results.json`, a statistical report with sample sizes, test statistics, raw and BH-adjusted p-values, effect sizes with intervals, and figures. |
| `data/` | `raw/` (registry snapshot), `interim/` (merged servers, findings, drift events, tool definitions), `processed/` (`master.csv` and reports). |
| `utils/` | I/O helpers, the statistics engine (Wilson intervals, Cliff's delta, chi-square with Cramér's V, bootstrap), and the report builder. |

Mapping to the paper: `rq1_threat_prevalence` and `rq5_ambient_authority`
feed RQ1 (prevalence and delegation risk); `rq2_tool_drift` feeds RQ2;
`rq3_popularity_security` feeds RQ3; `rq4_scanner_accuracy` and
`labeling/02_score_scanner.py` feed the scanner-validation subsection of the
Method and the precision-corrected prevalence in the Results.

## Reproduce the results from the shipped tables

No credentials are needed for this path. It re-runs validation, the codebook,
scanner scoring, and all five analyses on the data included here.

```bash
git clone <this-repo> && cd mcp-security-census
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py --analysis-only
```

Every result file and figure is regenerated under `analysis/rq*/` in under a
minute. Floating-point output is deterministic (bootstrap seed 42).

## Rebuild from the registry

The collection and processing stages need a GitHub token with public-repo
scope. The code fetch takes roughly eight hours and is resumable.

```bash
cp .env.example .env            # add GITHUB_TOKEN
python setup/setup_environment.py
python run_all.py               # or --skip-fetch to reuse an existing code cache
```

The registry changes weekly, so a fresh harvest will not reproduce the
August 2026 snapshot exactly; `data/raw/registry_servers.parquet` preserves
the snapshot the paper reports on.

## What was changed for public release

The paper reports results in aggregate and names no server. To keep the
artifact from serving as a target list, three finding-level files were
transformed before release:

- `data/interim/security_findings.csv` and both labeling sheets: server
  names and owner/repo identifiers are replaced by salted SHA-256 hashes
  (consistent across the three files), file paths are reduced to basenames,
  evidence strings are truncated to 60 characters, and the evidence of every
  `SEC_insecure_secrets` row is withheld. The `link` column of the labeling
  sheets is removed. The salt is not stored.
- `data/interim/code_index.csv` (the per-file fetch index with local paths)
  is not included; `data_collection/03_fetch_server_code.py` regenerates it.

All analyses run unchanged on the transformed files: the statistics depend
only on class, stratum, label, and the per-server columns of `master.csv`,
which carries registry names that are already public. Unredacted
finding-level rows are available to researchers on request for verification.

## Second-rater support

`labeling/02_score_scanner.py --second-rater <sheet.csv>` computes Cohen's
kappa against a second completed sheet; the protocol in `labeling_guide.txt`
is written to be applied independently.

## Citation

See `CITATION.cff`.

## License

Code: MIT. Data tables under `data/` and `labeling/`: CC BY 4.0. See `LICENSE`.
