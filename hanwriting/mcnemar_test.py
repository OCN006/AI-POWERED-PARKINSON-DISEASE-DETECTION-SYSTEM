"""
McNemar's test: baseline (EfficientNetB0) vs CBAM (EfficientNetB0+CBAM),
on pooled out-of-fold predictions across all 5 HandPD folds.

WHY POOLED OUT-OF-FOLD PREDICTIONS:
Each fold's validation set is a different, non-overlapping subset of
subjects. Running each fold's checkpoint on its OWN validation fold and
concatenating the results gives predictions covering the entire HandPD
dataset (736 images) with no image ever evaluated by a model that trained
on it. This is the standard way to get full-dataset predictions out of
k-fold CV, and gives McNemar's test much more statistical power than
testing any single fold alone (a handful of discordant pairs in one 150-
image fold is not enough to detect significance reliably).

McNemar's test itself only cares about the DISCORDANT pairs -- images where
the two models disagree (one right, one wrong). It ignores images both
models got right or both got wrong, since those don't inform which model
is better.

Usage:
    python mcnemar_test.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras
from statsmodels.stats.contingency_tables import mcnemar

from train_baseline import load_manifest, build_dataset, LABEL_TO_INT, OUTPUT_DIR


N_FOLDS = 5


def get_predictions_for_fold(checkpoint_path: Path, fold: int) -> pd.DataFrame:
    """Loads a checkpoint and returns per-image true label + predicted label
    for that fold's validation set."""
    model = keras.models.load_model(checkpoint_path, safe_mode=False)
    _, val_df = load_manifest(fold)
    val_ds = build_dataset(val_df, training=False)  # not shuffled -- order matches val_df

    probs = model.predict(val_ds, verbose=0).flatten()
    preds = (probs >= 0.5).astype(int)
    true_labels = val_df["label"].map(LABEL_TO_INT).values

    return pd.DataFrame({
        "image_path": val_df["image_path"].values,
        "subject_id": val_df["subject_id"].values,
        "fold": fold,
        "true_label": true_labels,
        "pred_prob": probs,
        "pred_label": preds,
        "correct": (preds == true_labels),
    })


def pool_predictions(model_prefix: str) -> pd.DataFrame:
    """Concatenates out-of-fold predictions across all 5 folds for one model."""
    all_folds = []
    for fold in range(N_FOLDS):
        checkpoint = OUTPUT_DIR / f"{model_prefix}_fold{fold}_best.keras"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"{checkpoint} not found -- make sure all 5 folds for "
                f"'{model_prefix}' finished training and saved a checkpoint."
            )
        print(f"  Predicting {model_prefix} fold {fold}...")
        all_folds.append(get_predictions_for_fold(checkpoint, fold))
    return pd.concat(all_folds, ignore_index=True)


def run_mcnemar(baseline_preds: pd.DataFrame, cbam_preds: pd.DataFrame):
    # Align by image_path to be safe (should already be in the same order
    # since both used the same fold splits, but this guards against any
    # ordering mismatch).
    merged = baseline_preds.merge(
        cbam_preds, on="image_path", suffixes=("_baseline", "_cbam")
    )
    assert (merged["true_label_baseline"] == merged["true_label_cbam"]).all(), \
        "Label mismatch between baseline and CBAM predictions -- fold splits may differ."

    both_correct = ((merged["correct_baseline"]) & (merged["correct_cbam"])).sum()
    baseline_only = ((merged["correct_baseline"]) & (~merged["correct_cbam"])).sum()
    cbam_only = ((~merged["correct_baseline"]) & (merged["correct_cbam"])).sum()
    both_wrong = ((~merged["correct_baseline"]) & (~merged["correct_cbam"])).sum()

    print(f"\nTotal images compared: {len(merged)}")
    print("\nContingency table:")
    print(f"                       CBAM correct   CBAM wrong")
    print(f"  Baseline correct     {both_correct:>13}   {baseline_only:>10}")
    print(f"  Baseline wrong       {cbam_only:>13}   {both_wrong:>10}")

    print(f"\nDiscordant pairs (what McNemar's test actually uses):")
    print(f"  Baseline right, CBAM wrong: {baseline_only}")
    print(f"  Baseline wrong, CBAM right: {cbam_only}")

    table = [[both_correct, baseline_only], [cbam_only, both_wrong]]
    result = mcnemar(table, exact=True)  # exact binomial test, appropriate for small discordant counts

    print(f"\nMcNemar's test statistic: {result.statistic:.4f}")
    print(f"p-value: {result.pvalue:.4f}")
    if result.pvalue < 0.05:
        print("-> Statistically significant difference between baseline and CBAM (p < 0.05).")
    else:
        print("-> No statistically significant difference detected (p >= 0.05).")
        print("   This is a legitimate, honest result to report -- it means CBAM's")
        print("   improvement in fold-to-fold consistency did not translate into a")
        print("   statistically detectable change in per-image correctness.")

    return merged


if __name__ == "__main__":
    print("Pooling baseline predictions across all folds...")
    baseline_preds = pool_predictions("baseline")
    print("\nPooling CBAM predictions across all folds...")
    cbam_preds = pool_predictions("cbam")

    merged = run_mcnemar(baseline_preds, cbam_preds)

    out_path = OUTPUT_DIR / "mcnemar_handpd_predictions.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nPer-image prediction comparison saved to: {out_path}")