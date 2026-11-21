"""
Aggregates 5-fold cross-validation results into a mean +/- std summary table.

This is meant to be reused for BOTH the baseline (EfficientNetB0) and the
CBAM model later, so the comparison table in your paper is built the same
way for each -- consistent methodology across your ablation.

Since train_baseline.py currently just prints metrics to the console rather
than saving them to a file, the simplest approach for now is to paste each
fold's final metrics into the FOLD_RESULTS dict below by hand. If you'd
rather this happen automatically, tell me and I'll modify train_baseline.py
to append results to a CSV after each fold -- more robust for when you have
many more runs to track (baseline x5 folds + CBAM x5 folds + any reruns).

Usage:
    python aggregate_results.py
"""

import numpy as np

# --- Paste your fold results here ---
# Each fold's dict should match the keys printed by train_baseline.py's
# final model.evaluate() output.

FOLD_RESULTS = {
    "baseline": [
        {"accuracy": 0.9211, "precision": 0.9821, "recall": 0.9167, "roc_auc": 0.9508, "pr_auc": 0.9868},  # fold 0
        {"accuracy": 0.6316, "precision": 0.8810, "recall": 0.6167, "roc_auc": 0.7091, "pr_auc": 0.8883},  # fold 1
        {"accuracy": 0.8472, "precision": 0.9298, "recall": 0.8833, "roc_auc": 0.8135, "pr_auc": 0.9514},  # fold 2
        {"accuracy": 0.8125, "precision": 1.0000, "recall": 0.7589, "roc_auc": 0.9509, "pr_auc": 0.9866},  # fold 3
        {"accuracy": 0.9583, "precision": 0.9914, "recall": 0.9583, "roc_auc": 0.9955, "pr_auc": 0.9991},  # fold 4
    ],
    # Add a "cbam": [...] entry here once the CBAM model's 5 folds are done,
    # using the same structure -- then this script will print both tables
    # side by side for an easy comparison.
}


def summarize(model_name: str, fold_results: list[dict]):
    metrics = fold_results[0].keys()
    print(f"\n=== {model_name} (n={len(fold_results)} folds) ===")
    print(f"{'Metric':<12} {'Mean':>8} {'Std':>8}   Per-fold values")
    for metric in metrics:
        values = np.array([r[metric] for r in fold_results])
        mean = values.mean()
        std = values.std(ddof=1)  # sample std (n-1), standard for reporting
        fold_str = ", ".join(f"{v:.3f}" for v in values)
        print(f"{metric:<12} {mean:>8.4f} {std:>8.4f}   [{fold_str}]")


if __name__ == "__main__":
    for model_name, results in FOLD_RESULTS.items():
        summarize(model_name, results)

    if len(FOLD_RESULTS) == 1:
        print("\n(Add a 'cbam' entry to FOLD_RESULTS once the CBAM model's "
              "folds are done to see both side by side.)")