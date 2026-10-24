"""
============================================================================
 stats.py, Statistical Tests + Effect Sizes for the MCP Census
============================================================================
 Project : Same Name, Different Server (MCP Security Census, IEEE BigData 2026)
 Purpose : One vetted implementation of every test the paper uses, so all
           five RQ scripts report statistics identically:
             - Wilson confidence intervals for prevalence (RQ1, RQ5)
             - chi-square / Fisher with Cramer's V and odds ratios
             - Mann-Whitney U with Cliff's delta (+ bootstrap CI)
             - Kruskal-Wallis with epsilon-squared, Dunn post-hoc
             - Spearman correlation with CI (RQ3)
             - logistic regression with odds ratios per predictor (RQ3)
             - precision / recall / F1 with Wilson CIs, Cohen's kappa (RQ4)
             - Benjamini-Hochberg adjustment
 Design  : prevalence data are binomial; magnitude data are heavy-tailed and
           zero-inflated, so distribution tests are non-parametric.
 Usage   : from utils.stats import *
============================================================================
"""

import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setup import config  # noqa: E402

RNG = np.random.default_rng(config.RANDOM_SEED)


# ---------------------------------------------------------------------------
# Prevalence (the backbone of RQ1 and RQ5)
# ---------------------------------------------------------------------------
def wilson_ci(k: int, n: int, ci: float = None) -> tuple:
    """Wilson score interval for a proportion. Correct at small k, unlike
    the normal approximation, which matters for rare threat classes."""
    ci = ci or config.CI_LEVEL
    if n == 0:
        return (float("nan"), float("nan"))
    z = sps.norm.ppf(1 - (1 - ci) / 2)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    lo, hi = centre - half, centre + half
    # clamp tiny floating-point residue so 0 events report exactly 0
    lo = 0.0 if lo < 1e-12 else lo
    hi = 1.0 if hi > 1 - 1e-12 else hi
    return (max(0.0, lo), min(1.0, hi))


def prevalence(k: int, n: int, label: str = "") -> dict:
    """Prevalence with Wilson CI, ready for a results table."""
    lo, hi = wilson_ci(k, n)
    return {"group": label, "k": int(k), "n": int(n),
            "prevalence": (k / n) if n else float("nan"),
            "ci_low": lo, "ci_high": hi}


def describe(x, label: str = "") -> dict:
    """n, mean, SD, median, IQR, min, max for one group (NaNs dropped)."""
    x = pd.Series(x).dropna().astype(float)
    if len(x) == 0:
        return {"group": label, "n": 0}
    q1, q3 = np.percentile(x, [25, 75])
    return {"group": label, "n": int(len(x)), "mean": float(x.mean()),
            "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "median": float(x.median()), "q1": float(q1), "q3": float(q3),
            "iqr": float(q3 - q1), "min": float(x.min()), "max": float(x.max())}


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------
def cliffs_delta(a, b) -> float:
    """Cliff's delta via U: delta = 2U/(n1*n2) - 1."""
    a = pd.Series(a).dropna().astype(float).values
    b = pd.Series(b).dropna().astype(float).values
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    u, _ = sps.mannwhitneyu(a, b, alternative="two-sided")
    return float(2.0 * u / (len(a) * len(b)) - 1.0)


def _band(value: float, bands) -> str:
    if np.isnan(value):
        return "undefined"
    for threshold, label in bands:
        if abs(value) < threshold:
            return label
    return bands[-1][1]


def cliffs_delta_magnitude(d: float) -> str:
    return _band(d, config.CLIFFS_DELTA_BANDS)


def cramers_v_magnitude(v: float) -> str:
    return _band(v, config.CRAMERS_V_BANDS)


def bootstrap_ci(stat_fn, *groups, n_boot=None, ci=None) -> tuple:
    """Seeded percentile bootstrap CI for any statistic."""
    n_boot = n_boot or config.N_BOOTSTRAP
    ci = ci or config.CI_LEVEL
    groups = [pd.Series(g).dropna().astype(float).values for g in groups]
    if any(len(g) < 2 for g in groups):
        return (float("nan"), float("nan"))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = stat_fn(*[g[RNG.integers(0, len(g), len(g))] for g in groups])
    lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Categorical comparisons
# ---------------------------------------------------------------------------
def chi_square(table) -> dict:
    """Chi-square with bias-corrected Cramer's V; Fisher fallback for small
    2x2 expected counts; odds ratio with CI for 2x2."""
    obs = np.asarray(table, dtype=float)
    chi2, p, dof, expected = sps.chi2_contingency(obs)
    n = obs.sum()
    r, c = obs.shape
    phi2 = max(0.0, chi2 / n - (r - 1) * (c - 1) / (n - 1))
    r_c = r - (r - 1) ** 2 / (n - 1)
    c_c = c - (c - 1) ** 2 / (n - 1)
    v = float(np.sqrt(phi2 / max(min(r_c - 1, c_c - 1), 1e-12)))
    out = {"test": "Chi-square", "chi2": float(chi2), "df": int(dof),
           "p": float(p), "n": int(n), "cramers_v": v,
           "v_magnitude": cramers_v_magnitude(v)}
    if obs.shape == (2, 2):
        if (expected < 5).any():
            _, p_f = sps.fisher_exact(obs)
            out.update({"test": "Fisher exact", "p": float(p_f)})
        a, b_, c_, d = obs.ravel()
        if 0 in (a, b_, c_, d):                       # Haldane-Anscombe
            a, b_, c_, d = a + .5, b_ + .5, c_ + .5, d + .5
        or_ = (a * d) / (b_ * c_)
        se = sqrt(1 / a + 1 / b_ + 1 / c_ + 1 / d)
        z = sps.norm.ppf(1 - (1 - config.CI_LEVEL) / 2)
        out.update({"odds_ratio": float(or_),
                    "or_ci_low": float(np.exp(np.log(or_) - z * se)),
                    "or_ci_high": float(np.exp(np.log(or_) + z * se))})
    return out


def two_proportion_test(k1, n1, k2, n2, label1="A", label2="B") -> dict:
    """Compare two prevalences (e.g. flagged rate in popular vs unpopular)."""
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    res = chi_square(table)
    res.update({"group_a": label1, "group_b": label2,
                **{f"prev_{label1}": k1 / n1 if n1 else float("nan"),
                   f"prev_{label2}": k2 / n2 if n2 else float("nan")}})
    a_lo, a_hi = wilson_ci(k1, n1)
    b_lo, b_hi = wilson_ci(k2, n2)
    res.update({"a_ci": (a_lo, a_hi), "b_ci": (b_lo, b_hi)})
    return res


# ---------------------------------------------------------------------------
# Distribution comparisons
# ---------------------------------------------------------------------------
def mann_whitney(a, b, label_a="A", label_b="B", boot_ci=True) -> dict:
    a_ = pd.Series(a).dropna().astype(float)
    b_ = pd.Series(b).dropna().astype(float)
    u, p = sps.mannwhitneyu(a_, b_, alternative="two-sided")
    d = cliffs_delta(a_, b_)
    out = {"test": "Mann-Whitney U", "group_a": label_a, "group_b": label_b,
           "n_a": int(len(a_)), "n_b": int(len(b_)), "U": float(u),
           "p": float(p), "cliffs_delta": d, "rank_biserial_r": d,
           "magnitude": cliffs_delta_magnitude(d)}
    if boot_ci:
        lo, hi = bootstrap_ci(cliffs_delta, a_, b_)
        out["delta_ci_low"], out["delta_ci_high"] = lo, hi
    return out


def kruskal_wallis(groups: dict) -> dict:
    labels = [l for l in groups
              if len(pd.Series(groups[l]).dropna()) > 0]
    arrays = [pd.Series(groups[l]).dropna().astype(float).values
              for l in labels]
    h, p = sps.kruskal(*arrays)
    n = sum(len(a) for a in arrays)
    k = len(arrays)
    eps = float((h - k + 1) / (n - k)) if n > k else float("nan")
    return {"test": "Kruskal-Wallis H", "groups": labels, "k": k,
            "n_total": int(n), "H": float(h), "df": k - 1, "p": float(p),
            "epsilon_squared": eps}


def dunn_posthoc(groups: dict) -> pd.DataFrame:
    """Dunn's test (tie-corrected) for all pairs, BH-adjusted."""
    labels = [l for l in groups if len(pd.Series(groups[l]).dropna()) > 0]
    data = pd.concat([pd.Series(groups[l]).dropna().astype(float)
                      .to_frame("value").assign(g=l) for l in labels])
    data["rank"] = sps.rankdata(data["value"])
    n = len(data)
    _, ties = np.unique(data["value"], return_counts=True)
    tie_term = ((ties ** 3 - ties).sum() / (12 * (n - 1))) if n > 1 else 0.0
    mr = data.groupby("g")["rank"].mean()
    ns = data.groupby("g")["rank"].count()
    rows = []
    for i, la in enumerate(labels):
        for lb in labels[i + 1:]:
            se = sqrt((n * (n + 1) / 12.0 - tie_term) * (1 / ns[la] + 1 / ns[lb]))
            z = (mr[la] - mr[lb]) / se if se > 0 else float("nan")
            rows.append({"group_a": la, "group_b": lb, "z": float(z),
                         "p_raw": float(2 * sps.norm.sf(abs(z))),
                         "cliffs_delta": cliffs_delta(groups[la], groups[lb])})
    df = pd.DataFrame(rows)
    if len(df):
        df["p_adj"] = adjust_pvalues(df["p_raw"].tolist())
        df["significant"] = df["p_adj"] < config.ALPHA
        df["magnitude"] = df["cliffs_delta"].map(cliffs_delta_magnitude)
    return df


# ---------------------------------------------------------------------------
# Association and modelling (RQ3)
# ---------------------------------------------------------------------------
def spearman(x, y) -> dict:
    """Spearman rho with Fisher-z CI. Rank-based, so the extreme star
    distribution (median 0, max >100k) needs no transformation."""
    s = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(s) < 4:
        return {"test": "Spearman", "n": len(s), "rho": float("nan"),
                "p": float("nan")}
    rho, p = sps.spearmanr(s["x"], s["y"])
    n = len(s)
    if abs(rho) >= 1.0:            # perfect rank agreement: CI degenerates
        return {"test": "Spearman", "n": int(n), "rho": float(rho),
                "p": float(p), "ci_low": float(rho), "ci_high": float(rho)}
    z = np.arctanh(rho)
    se = 1 / sqrt(n - 3)
    crit = sps.norm.ppf(1 - (1 - config.CI_LEVEL) / 2)
    return {"test": "Spearman", "n": int(n), "rho": float(rho), "p": float(p),
            "ci_low": float(np.tanh(z - crit * se)),
            "ci_high": float(np.tanh(z + crit * se))}


def logistic_regression(df: pd.DataFrame, outcome: str,
                        predictors: list) -> pd.DataFrame:
    """Logistic regression returning odds ratios with CIs per predictor."""
    import statsmodels.api as sm
    d = df[[outcome] + predictors].dropna()
    if d[outcome].nunique() < 2 or len(d) < 30:
        return pd.DataFrame()
    X = sm.add_constant(d[predictors].astype(float))
    model = sm.Logit(d[outcome].astype(int), X).fit(disp=False)
    conf = model.conf_int()
    out = pd.DataFrame({
        "term": model.params.index,
        "coef": model.params.values,
        "odds_ratio": np.exp(model.params.values),
        "or_ci_low": np.exp(conf[0].values),
        "or_ci_high": np.exp(conf[1].values),
        "p": model.pvalues.values,
    })
    out.attrs["n"] = int(len(d))
    out.attrs["pseudo_r2"] = float(model.prsquared)
    return out


# ---------------------------------------------------------------------------
# Scanner accuracy (RQ4)
# ---------------------------------------------------------------------------
def classification_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    """Precision, recall, F1, specificity, accuracy, each with Wilson CIs."""
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)
          if prec and rec and not np.isnan(prec) and not np.isnan(rec)
          else float("nan"))
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else float("nan")
    p_lo, p_hi = wilson_ci(tp, tp + fp)
    r_lo, r_hi = wilson_ci(tp, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "precision_ci_low": p_lo, "precision_ci_high": p_hi,
            "recall": rec, "recall_ci_low": r_lo, "recall_ci_high": r_hi,
            "f1": f1, "specificity": spec, "accuracy": acc,
            "n": tp + fp + fn + tn}


def cohens_kappa(a, b) -> dict:
    """Cohen's kappa for two label passes (inter-rater agreement)."""
    a = pd.Series(a).astype(int)
    b = pd.Series(b).astype(int)
    n = len(a)
    po = float((a == b).mean())
    pe = float(((a == 1).mean() * (b == 1).mean()) +
               ((a == 0).mean() * (b == 0).mean()))
    k = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return {"n": int(n), "observed_agreement": po,
            "expected_agreement": pe, "kappa": float(k)}


# ---------------------------------------------------------------------------
# Multiplicity + formatting
# ---------------------------------------------------------------------------
def adjust_pvalues(pvals: list, method: str = None) -> list:
    if not pvals:
        return []
    _, p_adj, _, _ = multipletests(pvals, alpha=config.ALPHA,
                                   method=method or config.P_ADJUST_METHOD)
    return [float(p) for p in p_adj]


def format_p(p: float) -> str:
    """APA-style: p < .001 or p = .034 (no leading zero)."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "p = n/a"
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")


def format_pct(x: float, dp: int = 1) -> str:
    return "n/a" if x is None or np.isnan(x) else f"{x*100:.{dp}f}%"
