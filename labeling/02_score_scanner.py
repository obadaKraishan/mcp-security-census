"""
============================================================================
 02_score_scanner.py, Score the Scanner Against Manual Labels (RQ4)
============================================================================
 Project : Same Name, Different Server (MCP Security Census, IEEE BigData 2026)
 Purpose : Turn the completed labelling sheet into the accuracy evidence the
           paper needs: precision, recall and F1 per threat class with Wilson
           intervals, a confusion matrix, and prevalence estimates corrected
           for measured precision. Optionally computes Cohen's kappa when a
           second rater's labels are present.
 Input   : labeling/labeling_sheet_completed.csv
 Output  : labeling/scanner_accuracy.csv / .json
           labeling/scanner_accuracy.txt   (APA/IEEE style report)
 Usage   : python labeling/02_score_scanner.py
           python labeling/02_score_scanner.py --second-rater labels2.csv
============================================================================
 Recall is estimated from the unflagged stratum: among sampled files with no
 finding, the share that a human judges to contain a real issue estimates the
 false-negative rate. Sampling is stratified, so this is an estimate with an
 interval, not a census figure, and the report says so.
============================================================================
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setup import config                                          # noqa: E402
from utils.io_helpers import save_csv, save_json, save_txt        # noqa: E402
from utils.stats import (classification_metrics, wilson_ci,       # noqa: E402
                         cohens_kappa, format_p)


def load_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"[FAIL] {path} not found.\n"
                 "  Fill in the 'label' column of labeling_sheet.csv and save "
                 f"it as {path.name}")
    df = pd.read_csv(path, low_memory=False)
    if "label" not in df.columns:
        sys.exit("[FAIL] sheet has no 'label' column")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    done = df[df["label"].isin([0, 1])]
    if not len(done):
        sys.exit("[FAIL] no rows labelled 0 or 1 yet")
    return df, done


def main():
    ap = argparse.ArgumentParser(description="Score the scanner")
    ap.add_argument("--second-rater", type=str, default=None,
                    help="CSV with a second pass for inter-rater agreement")
    args = ap.parse_args()

    print("=" * 76)
    print(" SCANNER ACCURACY — RQ4")
    print("=" * 76)
    df, done = load_labels(config.FILE_LABEL_DONE)
    print(f"  sheet rows              : {len(df):,}")
    print(f"  labelled                : {len(done):,} "
          f"({len(done)/len(df):.0%})")
    if len(done) < len(df):
        print(f"  [WARN] {len(df)-len(done):,} rows still unlabelled; results "
              "below use labelled rows only")

    flagged = done[done["stratum"] == "flagged"]
    unflagged = done[done["stratum"] == "unflagged"]
    print(f"  flagged stratum         : {len(flagged):,} (precision)")
    print(f"  unflagged stratum       : {len(unflagged):,} (recall)")

    # ---- recall from the unflagged stratum ---------------------------------
    # A labelled-1 unflagged row is a miss: a real issue the scanner did not
    # flag. miss_rate estimates P(miss | file has an issue but no finding).
    n_unf = len(unflagged)
    n_missed = int(unflagged["label"].sum()) if n_unf else 0
    miss_lo, miss_hi = wilson_ci(n_missed, n_unf) if n_unf else (float("nan"),) * 2

    # ---- per-class precision ----------------------------------------------
    rows = []
    for cls in config.THREAT_CLASSES:
        sub = flagged[flagged["threat_class"] == cls]
        if not len(sub):
            continue
        tp = int(sub["label"].sum())
        fp = int((sub["label"] == 0).sum())
        # class-attributed misses, where the rater named the class in notes
        fn = int(unflagged.apply(
            lambda r: r["label"] == 1 and cls.split("_")[0].lower()
            in str(r.get("notes", "")).lower(), axis=1).sum()) if n_unf else 0
        m = classification_metrics(tp=tp, fp=fp, fn=fn,
                                   tn=max(n_unf - n_missed, 0))
        m.update({"class": cls,
                  "severity": config.THREAT_CLASSES[cls]["severity"],
                  "n_labelled": len(sub)})
        rows.append(m)
        print(f"    {cls:<30s} P={m['precision']:.3f} "
              f"[{m['precision_ci_low']:.3f}, {m['precision_ci_high']:.3f}]  "
              f"(n={len(sub)})")

    acc = pd.DataFrame(rows)
    if acc.empty:
        sys.exit("[FAIL] no labelled flagged rows to score")

    # ---- pooled precision ---------------------------------------------------
    tp_all = int(flagged["label"].sum())
    fp_all = int((flagged["label"] == 0).sum())
    p_lo, p_hi = wilson_ci(tp_all, tp_all + fp_all)
    pooled_precision = tp_all / max(tp_all + fp_all, 1)

    # ---- corrected prevalence ------------------------------------------------
    corrected = []
    if config.FILE_MASTER.exists():
        master = pd.read_csv(config.FILE_MASTER, low_memory=False)
        scanned = master[master["has_code"].astype(bool)]
        n_scan = len(scanned)
        for _, r in acc.iterrows():
            col = f"has_{r['class']}"
            if col not in scanned.columns:
                continue
            k = int(scanned[col].fillna(0).sum())
            raw = k / n_scan if n_scan else float("nan")
            corrected.append({
                "class": r["class"], "k_flagged": k, "n_scanned": n_scan,
                "raw_prevalence": raw,
                "precision": r["precision"],
                "corrected_prevalence": raw * r["precision"],
                "corrected_low": raw * r["precision_ci_low"],
                "corrected_high": raw * r["precision_ci_high"],
            })

    # ---- inter-rater agreement -----------------------------------------------
    kappa = None
    if args.second_rater:
        p2 = Path(args.second_rater)
        if p2.exists():
            d2 = pd.read_csv(p2, low_memory=False)
            merged = done.merge(d2[["row_id", "label"]], on="row_id",
                                suffixes=("", "_r2"))
            merged = merged[merged["label_r2"].isin([0, 1])]
            if len(merged) >= 10:
                kappa = cohens_kappa(merged["label"], merged["label_r2"])
                print(f"  inter-rater kappa       : {kappa['kappa']:.3f} "
                      f"on {kappa['n']} shared rows")

    # ---- outputs ---------------------------------------------------------------
    save_csv(acc, config.FILE_SCANNER_SCORES, "scanner accuracy")
    save_json({"generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "n_labelled": int(len(done)),
               "pooled_precision": pooled_precision,
               "pooled_precision_ci": [p_lo, p_hi],
               "unflagged_sampled": n_unf,
               "unflagged_missed": n_missed,
               "miss_rate_ci": [miss_lo, miss_hi],
               "per_class": acc.to_dict("records"),
               "corrected_prevalence": corrected,
               "kappa": kappa},
              config.LABELING_DIR / "scanner_accuracy.json", "accuracy json")

    W = 78
    t = ["=" * W, " RQ4 — SCANNER ACCURACY AGAINST MANUAL LABELS",
         f" Snapshot : {config.SNAPSHOT_LABEL}",
         f" Generated: {datetime.now(timezone.utc).isoformat()}", "=" * W, "",
         " SAMPLE", "-" * W,
         f" Rows labelled                 : {len(done):,}",
         f"   flagged stratum (precision) : {len(flagged):,}",
         f"   unflagged stratum (recall)  : {n_unf:,}",
         f" Sampling                      : stratified by threat class, one",
         f"                                 finding per server, seed {config.RANDOM_SEED}",
         "", " POOLED PRECISION", "-" * W,
         f" True findings                 : {tp_all:,}",
         f" False positives               : {fp_all:,}",
         f" Precision                     : {pooled_precision:.3f}, "
         f"95% CI [{p_lo:.3f}, {p_hi:.3f}]", "",
         " PER-CLASS ACCURACY", "-" * W,
         f" {'Class':<28s} {'n':>4s} {'TP':>4s} {'FP':>4s} {'Prec':>7s} "
         f"{'95% CI':>18s}"]
    for _, r in acc.iterrows():
        t.append(f" {r['class']:<28s} {int(r['n_labelled']):>4d} "
                 f"{int(r['tp']):>4d} {int(r['fp']):>4d} "
                 f"{r['precision']:>7.3f} "
                 f"[{r['precision_ci_low']:.3f}, {r['precision_ci_high']:.3f}]")
    t += ["", " MISSED FINDINGS (recall stratum)", "-" * W,
          f" Unflagged files inspected     : {n_unf:,}",
          f" Judged to contain an issue    : {n_missed:,}",
          f" Miss rate                     : "
          f"{(n_missed/n_unf if n_unf else float('nan')):.3f}, "
          f"95% CI [{miss_lo:.3f}, {miss_hi:.3f}]",
          "",
          " Recall cannot be computed as a population figure because the true",
          " number of issues in unscanned code is unknown. The miss rate above",
          " is the estimated share of unflagged files that a rater judged to",
          " contain an issue, and bounds how much the scanner underestimates."]
    if corrected:
        t += ["", " PRECISION-CORRECTED PREVALENCE", "-" * W,
              f" {'Class':<28s} {'raw':>8s} {'corrected':>10s} {'95% CI':>20s}"]
        for c in corrected:
            t.append(f" {c['class']:<28s} {c['raw_prevalence']*100:>7.2f}% "
                     f"{c['corrected_prevalence']*100:>9.2f}% "
                     f"    [{c['corrected_low']*100:5.2f}, "
                     f"{c['corrected_high']*100:5.2f}]")
        t += ["",
              " Corrected prevalence multiplies the observed rate by measured",
              " precision, giving a conservative estimate of true prevalence."]
    if kappa:
        t += ["", " INTER-RATER AGREEMENT", "-" * W,
              f" Shared rows                   : {kappa['n']:,}",
              f" Observed agreement            : {kappa['observed_agreement']:.3f}",
              f" Cohen's kappa                 : {kappa['kappa']:.3f}"]
    t += ["", " READY-TO-PASTE (IEEE prose)", "-" * W,
          f' "Across {len(flagged):,} manually labelled findings, the scanner',
          f'  achieved a pooled precision of {pooled_precision:.2f} '
          f'(95% CI [{p_lo:.2f}, {p_hi:.2f}]). Precision varied by class',
          f'  (Table X). Among {n_unf:,} sampled files with no finding, '
          f'{n_missed:,} were',
          f'  judged to contain an issue the scanner missed, an estimated miss',
          f'  rate of {(n_missed/n_unf if n_unf else 0):.2f} '
          f'(95% CI [{miss_lo:.2f}, {miss_hi:.2f}])."', "", "=" * W]
    save_txt("\n".join(t), config.LABELING_DIR / "scanner_accuracy.txt",
             "accuracy report")

    print("-" * 76)
    print(f"  pooled precision        : {pooled_precision:.3f} "
          f"[{p_lo:.3f}, {p_hi:.3f}]")
    print(f"  miss rate (unflagged)   : "
          f"{(n_missed/n_unf if n_unf else float('nan')):.3f}")
    print(f"  report                  : "
          f"{config.LABELING_DIR / 'scanner_accuracy.txt'}")
    print("  Next: run the RQ analyses in analysis/")


if __name__ == "__main__":
    main()
