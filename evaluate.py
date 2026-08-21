import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from main import BidslabSleepFusionDataset, DualSensorFusionModel, create_subject_level_split, get_device


CLASS_NAMES = ["Wake", "N1", "N2", "N3", "REM", "Unscored"]
NUM_CLASSES = len(CLASS_NAMES)
DEFAULT_DATA_PATH = os.path.join(
    "Sleepdata",
    "a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the six-class sleep staging model.")
    parser.add_argument("--checkpoint", default="bidslab_sleep_model.pth", help="Model checkpoint path.")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Root directory containing the dataset.")
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size (default: 128).")
    parser.add_argument("--seq-len", type=int, default=21, help="Sequence length for context window.")
    parser.add_argument("--output", help="Optional path for a JSON metrics report.")
    return parser.parse_args()


def calculate_metrics(targets, predictions, probabilities):
    labels = list(range(NUM_CLASSES))
    report = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )
    result = {
        "accuracy": float(report["accuracy"]),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist(),
        "class_names": CLASS_NAMES,
    }
    try:
        result["macro_roc_auc"] = float(
            roc_auc_score(
                targets,
                probabilities,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        result["macro_roc_auc"] = None
    return result


def print_metrics(metrics):
    print("\n" + "=" * 60)
    print("SIX-CLASS SLEEP STAGING EVALUATION")
    print("=" * 60)
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print("\nPrecision, recall, and F1-score:")
    report = metrics["classification_report"]
    print(f"{'Class':<12}{'Precision':>12}{'Recall':>12}{'F1-score':>12}{'Support':>12}")
    for class_name in CLASS_NAMES:
        scores = report[class_name]
        print(
            f"{class_name:<12}{scores['precision']:>12.4f}{scores['recall']:>12.4f}"
            f"{scores['f1-score']:>12.4f}{int(scores['support']):>12}"
        )
    for average_name in ("macro avg", "weighted avg"):
        scores = report[average_name]
        print(
            f"{average_name:<12}{scores['precision']:>12.4f}{scores['recall']:>12.4f}"
            f"{scores['f1-score']:>12.4f}{int(scores['support']):>12}"
        )
    roc_auc = metrics["macro_roc_auc"]
    print(f"\nMacro ROC-AUC (one-vs-rest): {'unavailable' if roc_auc is None else f'{roc_auc:.4f}'}")
    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(np.asarray(metrics["confusion_matrix"]))


def evaluate_model(checkpoint_path, data_path, batch_size, seq_len=21):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Dataset directory not found: {data_path}")
    if batch_size < 1:
        raise ValueError("--batch-size must be a positive integer")

    dataset = BidslabSleepFusionDataset(root_dir=data_path, seq_len=seq_len)
    if len(dataset) == 0:
        raise ValueError("No valid labeled epochs were found in the dataset directory.")

    _, test_dataset = create_subject_level_split(dataset)
    if len(test_dataset) == 0:
        raise ValueError("The subject-level test split is empty.")

    device = get_device()
    model = DualSensorFusionModel(num_classes=NUM_CLASSES).to(device)
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    except RuntimeError as error:
        raise ValueError(f"Checkpoint shape does not match the six-class model: {error}") from error
    except (OSError, RuntimeError, KeyError) as error:
        raise ValueError(f"Could not load checkpoint '{checkpoint_path}': {error}") from error

    model.eval()
    targets, predictions, probabilities = [], [], []
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Evaluating {len(test_dataset)} held-out epochs from the subject-level test split...")
    with torch.no_grad():
        for hr_batch, motion_batch, labels in loader:
            outputs = model(hr_batch.to(device), motion_batch.to(device))
            probabilities.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            targets.extend(labels.numpy())

    return calculate_metrics(np.asarray(targets), np.asarray(predictions), np.asarray(probabilities))


def main():
    args = parse_args()
    try:
        metrics = evaluate_model(args.checkpoint, args.data_path, args.batch_size, args.seq_len)
        print_metrics(metrics)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as output_file:
                json.dump(metrics, output_file, indent=2)
            print(f"\nJSON report saved to: {args.output}")
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())