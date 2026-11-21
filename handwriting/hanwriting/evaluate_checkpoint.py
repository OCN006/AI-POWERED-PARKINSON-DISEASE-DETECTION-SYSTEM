"""
Reload a saved checkpoint and re-print its validation metrics -- useful when
you trained successfully but lost the printed output (didn't copy it,
terminal closed, etc.). Much faster than retraining since this just loads
weights and runs one evaluation pass.

Usage:
    python evaluate_checkpoint.py --checkpoint outputs/cbam_fold1_best.keras --fold 1
"""

import argparse
from tensorflow import keras

from train_baseline import load_manifest, build_dataset


def main(checkpoint_path: str, fold: int):
    print(f"Loading {checkpoint_path} ...")
    model = keras.models.load_model(checkpoint_path, safe_mode=False)

    _, val_df = load_manifest(fold)
    val_ds = build_dataset(val_df, training=False)

    print(f"Re-evaluating on fold {fold} validation set "
          f"({len(val_df)} images, {val_df['subject_id'].nunique()} subjects)...")
    metrics = model.evaluate(val_ds, return_dict=True)

    print(f"\nRecovered validation metrics (fold {fold}):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to the saved .keras checkpoint")
    parser.add_argument("--fold", type=int, required=True, help="Which fold this checkpoint was validated on")
    args = parser.parse_args()
    main(args.checkpoint, args.fold)