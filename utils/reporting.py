"""
============================================================================
 reporting.py, APA-7 / IEEE Statistical Report Generator (.txt)
============================================================================
 Project : Same Name, Different Server (MCP Security Census, IEEE BigData 2026)
 Purpose : Turn the dicts and DataFrames produced by utils.stats into the
           rqN_apa_report.txt files: population and subsample sizes,
           prevalence tables with Wilson CIs, test statistics, exact and
           BH-adjusted p-values, effect sizes with intervals, scanner
           accuracy tables, and paste-ready IEEE sentences.
 Usage   : from utils.reporting import ReportBuilder
============================================================================
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setup import config                                    # noqa: E402
from utils.stats import format_p, format_pct                # noqa: E402

W = 78


class ReportBuilder:
    """Accumulates sections, then renders one APA/IEEE .txt report."""

    def __init__(self, rq_key: str):
        rq = config.RESEARCH_QUESTIONS[rq_key]
        self.rq_key = rq_key
        self.lines: list[str] = []
        self.lines += [
            "=" * W,
            f" {rq['title']}",
            "=" * W,
            f" Research question : {rq['question']}",
            f" Registry snapshot : {config.SNAPSHOT_LABEL}",
            f" Generated (UTC)   : {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}",
            f" Alpha             : {config.ALPHA}   |   CI: "
            f"{int(config.CI_LEVEL*100)}%   |   p-adjustment: Benjamini-Hochberg",
            f" Bootstrap draws   : {config.N_BOOTSTRAP:,} "
            f"(seed {config.RANDOM_SEED})",
            "=" * W, "",
        ]

    # ---- structure ---------------------------------------------------------
    def section(self, title: str):
        self.lines += ["", "-" * W, f" {title}", "-" * W]

    def text(self, *paragraphs: str):
        for p in paragraphs:
            self.lines.append(f" {p}")

    def population(self, **counts):
        """Record the population denominators that frame every result."""
        self.section("Population and analysis sets")
        for k, v in counts.items():
            label = k.replace("_", " ")
            self.lines.append(f" {label:<44s} {v:>12,}"
                              if isinstance(v, (int, np.integer))
                              else f" {label:<44s} {v:>12}")

    # ---- tables ------------------------------------------------------------
    def prevalence_table(self, rows: list, title: str):
        """k/n prevalence with Wilson CIs, one row per class or group."""
        self.section(title)
        hdr = (f" {'Class / group':<30s} {'k':>7s} {'n':>8s} {'prev':>8s} "
               f"{'95% CI':>18s}")
        self.lines += [hdr, " " + "-" * (len(hdr) - 1)]
        for r in rows:
            ci = f"[{r['ci_low']*100:5.2f}, {r['ci_high']*100:5.2f}]"
            self.lines.append(
                f" {str(r['group'])[:30]:<30s} {r['k']:>7,} {r['n']:>8,} "
                f"{format_pct(r['prevalence'], 2):>8s} {ci:>18s}")

    def descriptives_table(self, rows: list, metric: str):
        self.section(f"Descriptive statistics — {metric}")
        hdr = (f" {'Group':<22s} {'n':>7s} {'Mean':>10s} {'SD':>10s} "
               f"{'Median':>9s} {'Q1':>8s} {'Q3':>8s}")
        self.lines += [hdr, " " + "-" * (len(hdr) - 1)]
        for d in rows:
            if d.get("n", 0) == 0:
                self.lines.append(f" {d['group']:<22s} {'0':>7s}   (no data)")
                continue
            self.lines.append(
                f" {str(d['group'])[:22]:<22s} {d['n']:>7,} {d['mean']:>10.3f} "
                f"{d['sd']:>10.3f} {d['median']:>9.3f} {d['q1']:>8.3f} "
                f"{d['q3']:>8.3f}")

    def dataframe_table(self, df: pd.DataFrame, title: str,
                        float_fmt: str = "{:.4f}"):
        self.section(title)
        if df is None or df.empty:
            self.lines.append(" (no rows)")
            return
        show = df.copy()
        for c in show.columns:
            if pd.api.types.is_float_dtype(show[c]):
                show[c] = show[c].map(
                    lambda v: float_fmt.format(v) if pd.notna(v) else "—")
        self.lines += [" " + row for row in
                       show.to_string(index=False).split("\n")]

    # ---- test results ------------------------------------------------------
    def chi_square_result(self, r: dict, what: str, p_adj: float = None):
        self.section(f"Categorical test — {what}")
        self.text(f"Test          : {r['test']}",
                  f"n             : {r['n']:,}",
                  f"Statistic     : chi2({r.get('df','-')}) = {r['chi2']:.2f}",
                  f"p (raw)       : {format_p(r['p'])}")
        if p_adj is not None:
            self.text(f"p (BH-adj.)   : {format_p(p_adj)}")
        self.text(f"Effect size   : Cramer's V = {r['cramers_v']:.3f} "
                  f"({r.get('v_magnitude','')})")
        if "odds_ratio" in r:
            self.text(f"Odds ratio    : OR = {r['odds_ratio']:.3f}, "
                      f"{int(config.CI_LEVEL*100)}% CI "
                      f"[{r['or_ci_low']:.3f}, {r['or_ci_high']:.3f}]")
        p_show = format_p(p_adj if p_adj is not None else r["p"])
        sig = (p_adj if p_adj is not None else r["p"]) < config.ALPHA
        or_txt = (f", OR = {r['odds_ratio']:.2f} "
                  f"[{r['or_ci_low']:.2f}, {r['or_ci_high']:.2f}]"
                  if "odds_ratio" in r else "")
        self.text("", "READY-TO-PASTE (IEEE prose):",
                  f'"{what} {"differed significantly" if sig else "did not differ significantly"} '
                  f'(chi2({r.get("df","-")}) = {r["chi2"]:.2f}, {p_show}, '
                  f'Cramer\'s V = {r["cramers_v"]:.2f}{or_txt})."')

    def mann_whitney_result(self, r: dict, metric: str, p_adj: float = None):
        self.section(f"Distribution test — {metric}: "
                     f"{r['group_a']} vs {r['group_b']}")
        self.text(f"Test          : {r['test']} (two-sided)",
                  f"Sample sizes  : n({r['group_a']}) = {r['n_a']:,}, "
                  f"n({r['group_b']}) = {r['n_b']:,}",
                  f"Statistic     : U = {r['U']:,.1f}",
                  f"p (raw)       : {format_p(r['p'])}")
        if p_adj is not None:
            self.text(f"p (BH-adj.)   : {format_p(p_adj)}")
        ci = ""
        if "delta_ci_low" in r and not np.isnan(r["delta_ci_low"]):
            ci = (f", {int(config.CI_LEVEL*100)}% CI "
                  f"[{r['delta_ci_low']:.3f}, {r['delta_ci_high']:.3f}]")
        self.text(f"Effect size   : Cliff's delta = {r['cliffs_delta']:.3f}{ci} "
                  f"({r['magnitude']})")
        p_show = format_p(p_adj if p_adj is not None else r["p"])
        direction = "higher" if r["cliffs_delta"] > 0 else "lower"
        self.text("", "READY-TO-PASTE (IEEE prose):",
                  f'"{r["group_a"]} showed {direction} {metric} than '
                  f'{r["group_b"]} (Mann-Whitney U = {r["U"]:,.1f}, {p_show}, '
                  f'Cliff\'s delta = {r["cliffs_delta"]:.2f}{ci}), a '
                  f'{r["magnitude"]} effect."')

    def kruskal_result(self, r: dict, metric: str):
        self.section(f"Omnibus test — {metric} across {r['k']} groups")
        self.text(f"Test          : {r['test']}",
                  f"Groups (k)    : {r['k']} "
                  f"({', '.join(map(str, r['groups']))})",
                  f"Total n       : {r['n_total']:,}",
                  f"Statistic     : H({r['df']}) = {r['H']:.2f}",
                  f"p             : {format_p(r['p'])}",
                  f"Effect size   : epsilon-squared = "
                  f"{r['epsilon_squared']:.4f}")
        sig = r["p"] < config.ALPHA
        self.text("", "READY-TO-PASTE (IEEE prose):",
                  f'"A Kruskal-Wallis test '
                  f'{"indicated a significant difference in" if sig else "found no significant difference in"} '
                  f'{metric} across the {r["k"]} groups '
                  f'(H({r["df"]}) = {r["H"]:.2f}, {format_p(r["p"])}'
                  f'{f", epsilon-squared = {r['epsilon_squared']:.3f}" if sig else ""})."')

    def spearman_result(self, r: dict, what: str):
        self.section(f"Association — {what}")
        self.text(f"Test          : Spearman rank correlation",
                  f"n             : {r['n']:,}",
                  f"rho           : {r['rho']:.3f}, "
                  f"{int(config.CI_LEVEL*100)}% CI "
                  f"[{r.get('ci_low', float('nan')):.3f}, "
                  f"{r.get('ci_high', float('nan')):.3f}]",
                  f"p             : {format_p(r['p'])}")
        self.text("", "READY-TO-PASTE (IEEE prose):",
                  f'"{what} showed a '
                  f'{"significant" if r["p"] < config.ALPHA else "non-significant"} '
                  f'rank correlation (rho = {r["rho"]:.2f}, '
                  f'{format_p(r["p"])}, n = {r["n"]:,})."')

    def logistic_result(self, df: pd.DataFrame, outcome: str):
        self.section(f"Logistic regression — predicting {outcome}")
        if df is None or df.empty:
            self.lines.append(" (model did not converge or too few rows)")
            return
        self.text(f"Observations  : {df.attrs.get('n', 'n/a'):,}",
                  f"Pseudo R^2    : {df.attrs.get('pseudo_r2', float('nan')):.4f}")
        hdr = (f" {'Term':<24s} {'OR':>9s} {'95% CI':>22s} {'p':>10s}")
        self.lines += ["", hdr, " " + "-" * (len(hdr) - 1)]
        for _, r in df.iterrows():
            ci = f"[{r['or_ci_low']:.3f}, {r['or_ci_high']:.3f}]"
            self.lines.append(f" {str(r['term'])[:24]:<24s} "
                              f"{r['odds_ratio']:>9.3f} {ci:>22s} "
                              f"{format_p(r['p']):>10s}")

    def classification_result(self, rows: list, title: str):
        """Scanner accuracy table (RQ4): per-class precision/recall/F1."""
        self.section(title)
        hdr = (f" {'Class':<28s} {'TP':>5s} {'FP':>5s} {'FN':>5s} "
               f"{'Prec':>7s} {'Recall':>7s} {'F1':>7s}")
        self.lines += [hdr, " " + "-" * (len(hdr) - 1)]
        for r in rows:
            self.lines.append(
                f" {str(r.get('class', r.get('group','')))[:28]:<28s} "
                f"{r['tp']:>5,} {r['fp']:>5,} {r['fn']:>5,} "
                f"{r['precision']:>7.3f} {r['recall']:>7.3f} {r['f1']:>7.3f}")
        self.lines.append("")
        for r in rows:
            self.lines.append(
                f" {str(r.get('class', r.get('group','')))[:28]:<28s} "
                f"precision 95% CI [{r['precision_ci_low']:.3f}, "
                f"{r['precision_ci_high']:.3f}]   recall 95% CI "
                f"[{r['recall_ci_low']:.3f}, {r['recall_ci_high']:.3f}]")

    # ---- boilerplate sections ---------------------------------------------
    def methods_note(self):
        self.section("Statistical approach and assumptions")
        self.text(
            "Prevalence estimates use Wilson score intervals, which remain",
            "accurate for rare classes where the normal approximation fails.",
            "Counts of findings per server are heavy-tailed and zero-inflated,",
            "so distribution comparisons use non-parametric rank tests",
            "(Mann-Whitney, Kruskal-Wallis) with Cliff's delta and",
            "epsilon-squared effect sizes. Popularity is analysed with rank",
            "methods because star counts span several orders of magnitude.",
            "Families of related tests are corrected with Benjamini-Hochberg",
            f"at FDR {config.ALPHA}; adjusted p-values are reported alongside",
            f"raw ones. Bootstrap intervals use {config.N_BOOTSTRAP:,} "
            f"resamples under seed {config.RANDOM_SEED}.")

    def detection_caveat(self):
        self.section("Detection caveat")
        self.text(
            "Findings are produced by pattern-based static analysis of server",
            "metadata and source code. They are risk indicators, not confirmed",
            "vulnerabilities: the scanner cannot establish reachability or",
            "exploitability. Precision and recall against a manually labelled",
            "sample are reported in RQ4, and results are presented in",
            "aggregate without naming individual servers.")

    def render(self) -> str:
        return "\n".join(self.lines + ["", "=" * W, " END OF REPORT", "=" * W])
