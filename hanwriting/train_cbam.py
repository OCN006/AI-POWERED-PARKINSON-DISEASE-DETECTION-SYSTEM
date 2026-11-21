"""
EfficientNetB0 + CBAM training script.

Reuses the EXACT data pipeline, class weighting, optimizer, and callback
protocol from train_baseline.py -- the only difference is the model
architecture (CBAM inserted after the backbone's feature map, before
pooling). This keeps the ablation comparison valid: any performance
difference is attributable to CBAM, not to a changed training setup.

Usage:
    python train_cbam.py --fold 0
    python train_cbam.py --fold 0 --check_data
"""

import argparse
from pathlib import Path

from tensorflow import keras
from tensorflow.keras.applications.efficientnet import EfficientNetB0

from cbam import cbam_block
# Reuse everything else directly from the baseline script -- same data
# pipeline, same class weighting, same optimizer helper.
from train_baseline import (
    IMG_SIZE, HEAD_EPOCHS, FINETUNE_EPOCHS, HEAD_LR, FINETUNE_LR,
    UNFREEZE_FRACTION, OUTPUT_DIR,
    load_manifest, build_dataset, get_class_weights, get_optimizer,
    unfreeze_last_fraction, check_data,
)


def build_model_with_cbam() -> tuple[keras.Model, keras.Model]:
    """
    Same as train_baseline.build_model(), except:
    - backbone pooling=None (we need the raw 7x7x1280 feature map to apply
      CBAM to, not the already-pooled 1280-d vector)
    - CBAM block inserted after the backbone, before GlobalAveragePooling2D
    """
    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling=None,  # keep spatial dims for CBAM to operate on
    )
    backbone.trainable = False  # frozen for phase 1

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = backbone(inputs, training=False)          # (batch, 7, 7, 1280)
    x = cbam_block(x, reduction_ratio=16, kernel_size=7, name="cbam")
    x = keras.layers.GlobalAveragePooling2D(name="post_cbam_pool")(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="efficientnetb0_cbam")
    return model, backbone


def train(fold: int):
    OUTPUT_DIR.mkdir(exist_ok=True)
    checkpoint_path = OUTPUT_DIR / f"cbam_fold{fold}_best.keras"

    train_df, val_df = load_manifest(fold)
    print(f"Train: {len(train_df)} images ({train_df['subject_id'].nunique()} subjects) | "
          f"Val: {len(val_df)} images ({val_df['subject_id'].nunique()} subjects)")

    train_ds = build_dataset(train_df, training=True)
    val_ds = build_dataset(val_df, training=False)
    class_weights = get_class_weights(train_df)

    model, backbone = build_model_with_cbam()
    model.summary()  # worth inspecting once to confirm CBAM layers are present

    metrics = [
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="roc_auc", curve="ROC"),
        keras.metrics.AUC(name="pr_auc", curve="PR"),
    ]

    print("\n=== Phase 1: training head + CBAM (backbone frozen) ===")
    model.compile(optimizer=get_optimizer(HEAD_LR), loss="binary_crossentropy", metrics=metrics)
    model.fit(
        train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS, class_weight=class_weights,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=6, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max", factor=0.5, patience=3),
        ],
    )

    print("\n=== Phase 2: fine-tuning (last 25% of backbone unfrozen) ===")
    unfreeze_last_fraction(backbone, UNFREEZE_FRACTION)
    model.compile(optimizer=get_optimizer(FINETUNE_LR), loss="binary_crossentropy", metrics=metrics)
    model.fit(
        train_ds, validation_data=val_ds, epochs=FINETUNE_EPOCHS, class_weight=class_weights,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=8, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max", factor=0.5, patience=4),
            keras.callbacks.ModelCheckpoint(
                str(checkpoint_path), monitor="val_pr_auc", mode="max",
                save_best_only=True, save_weights_only=False,
            ),
        ],
    )

    print(f"\nBest checkpoint saved to: {checkpoint_path}")
    final_metrics = model.evaluate(val_ds, return_dict=True)
    print(f"\nFinal validation metrics (CBAM, fold {fold}):")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--check_data", action="store_true")
    args = parser.parse_args()

    if args.check_data:
        check_data(args.fold)
    else:
        train(args.fold)