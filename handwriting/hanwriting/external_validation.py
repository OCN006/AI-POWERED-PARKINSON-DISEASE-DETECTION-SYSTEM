"""
External validation on NewHandPD.

NewHandPD was NEVER used during training or fold splitting -- this is the
core novelty claim of the project. Since HandPD training produced 5
checkpoints per model (one per fold, each trained on a different ~80% of
HandPD), we evaluate ALL 5 checkpoints of each model on the SAME full
NewHandPD set and report mean +/- std -- this tells us how consistently
each fold's model generalizes to a genuinely different data source, not
just to held-out HandPD subjects.

For McNemar's test on the external set, we use each model's ENSEMBLE
prediction (majority vote across the 5 fold checkpoints) rather than
picking one arbitrary fold's checkpoint -- this is a fairer, less
arbitrary comparison than choosing e.g. "fold 0's checkpoint" by chance.

Usage:
    python external_validation.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    average_precision_score,
)

from train_baseline import build_dataset, LABEL_TO_INT, OUTPUT_DIR


N_FOLDS = 5
NEWHANDPD_MANIFEST = Path("data/manifests/newhandpd_manifest.csv")


def load_external_set() -> pd.DataFrame:
    if not NEWHANDPD_MANIFEST.exists():
        raise FileNotFoundError(f"{NEWHANDPD_MANIFEST} not found -- run data_pipeline.py first.")
    return pd.read_csv(NEWHANDPD_MANIFEST)


def predict_with_checkpoint(checkpoint_path: Path, df: pd.DataFrame) -> np.ndarray:
    model = keras.models.load_model(checkpoint_path, safe_mode=False)
    ds = build_dataset(df, training=False)
    probs = model.predict(ds, verbose=0).flatten()
    return probs


def evaluate_all_folds(model_prefix: str, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Returns:
      - per-fold metrics DataFrame (for mean +/- std reporting)
      - array of shape (n_folds, n_images) of predicted probabilities,
        for building the ensemble prediction used in McNemar's test
    """
    true_labels = df["label"].map(LABEL_TO_INT).values
    all_fold_probs = []
    rows = []

    for fold in range(N_FOLDS):
        checkpoint = OUTPUT_DIR / f"{model_prefix}_fold{fold}_best.keras"
        if not checkpoint.exists():
            raise FileNotFoundError(f"{checkpoint} not found.")
        print(f"  Evaluating {model_prefix} fold {fold} checkpoint on NewHandPD...")
        probs = predict_with_checkpoint(checkpoint, df)
        preds = (probs >= 0.5).astype(int)
        all_fold_probs.append(probs)

        rows.append({
            "fold": fold,
            "accuracy": accuracy_score(true_labels, preds),
            "precision": precision_score(true_labels, preds, zero_division=0),
            "recall": recall_score(true_labels, preds, zero_division=0),
            "roc_auc": roc_auc_score(true_labels, probs),
            "pr_auc": average_precision_score(true_labels, probs),
        })

    return pd.DataFrame(rows), np.array(all_fold_probs)


def summarize_external(model_name: str, fold_metrics: pd.DataFrame):
    print(f"\n=== {model_name} on NewHandPD (external test, n={len(fold_metrics)} fold-checkpoints) ===")
    print(f"{'Metric':<12} {'Mean':>8} {'Std':>8}   Per-fold values")
    for metric in ["accuracy", "precision", "recall", "roc_auc", "pr_auc"]:
        values = fold_metrics[metric].values
        fold_str = ", ".join(f"{v:.3f}" for v in values)
        print(f"{metric:<12} {values.mean():>8.4f} {values.std(ddof=1):>8.4f}   [{fold_str}]")


def ensemble_predictions(fold_probs: np.ndarray) -> np.ndarray:
    """Average probability across the 5 fold checkpoints, then threshold."""
    mean_probs = fold_probs.mean(axis=0)
    return (mean_probs >= 0.5).astype(int)


def run_mcnemar_external(df: pd.DataFrame, baseline_fold_probs, cbam_fold_probs):
    true_labels = df["label"].map(LABEL_TO_INT).values
    baseline_preds = ensemble_predictions(baseline_fold_probs)
    cbam_preds = ensemble_predictions(cbam_fold_probs)

    baseline_correct = (baseline_preds == true_labels)
    cbam_correct = (cbam_preds == true_labels)

    both_correct = (baseline_correct & cbam_correct).sum()
    baseline_only = (baseline_correct & ~cbam_correct).sum()
    cbam_only = (~baseline_correct & cbam_correct).sum()
    both_wrong = (~baseline_correct & ~cbam_correct).sum()

    print(f"\n=== McNemar's test on NewHandPD (ensemble predictions) ===")
    print(f"Total images: {len(df)}")
    print("\nContingency table:")
    print(f"                       CBAM correct   CBAM wrong")
    print(f"  Baseline correct     {both_correct:>13}   {baseline_only:>10}")
    print(f"  Baseline wrong       {cbam_only:>13}   {both_wrong:>10}")

    table = [[both_correct, baseline_only], [cbam_only, both_wrong]]
    result = mcnemar(table, exact=True)
    print(f"\nMcNemar's statistic: {result.statistic:.4f}")
    print(f"p-value: {result.pvalue:.4f}")
    if result.pvalue < 0.05:
        print("-> Statistically significant difference on external data (p < 0.05).")
    else:
        print("-> No statistically significant difference detected (p >= 0.05).")

    return pd.DataFrame({
        "image_path": df["image_path"].values,
        "true_label": true_labels,
        "baseline_ensemble_pred": baseline_preds,
        "cbam_ensemble_pred": cbam_preds,
        "baseline_correct": baseline_correct,
        "cbam_correct": cbam_correct,
    })


if __name__ == "__main__":
    df = load_external_set()
    print(f"NewHandPD external test set: {len(df)} images, {df['subject_id'].nunique()} subjects\n")

    print("Evaluating baseline checkpoints...")
    baseline_metrics, baseline_probs = evaluate_all_folds("baseline", df)
    print("\nEvaluating CBAM checkpoints...")
    cbam_metrics, cbam_probs = evaluate_all_folds("cbam", df)

    summarize_external("Baseline", baseline_metrics)
    summarize_external("CBAM", cbam_metrics)

    print("\n--- Generalization gap check ---")
    print("Compare these numbers to the HandPD in-distribution results:")
    print("If NewHandPD accuracy is notably lower than HandPD CV accuracy,")
    print("that gap IS your novelty finding -- report it honestly, it's the")
    print("point of doing external validation at all.")

    comparison_df = run_mcnemar_external(df, baseline_probs, cbam_probs)

    out_path = OUTPUT_DIR / "external_validation_predictions.csv"
    comparison_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")