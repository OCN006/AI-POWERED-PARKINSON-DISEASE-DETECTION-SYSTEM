"""
EfficientNetB0 baseline for Parkinson's handwriting classification.

This is the REFERENCE baseline (no CBAM yet) -- get this fully working and
stable before adding the attention module. Every later model (CBAM version)
will reuse this exact data pipeline and training protocol so the ablation
comparison is valid (only the architecture changes, nothing else).

Usage:
    python train_baseline.py --fold 0                 # train with fold 0 as validation
    python train_baseline.py --fold 0 --check_data     # just verify the data loads correctly, no training

Trains on HandPD only (5-fold CV, subject-stratified). NewHandPD is never
touched here -- external evaluation happens in a separate script once this
baseline (and later the CBAM model) is trained.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from sklearn.utils.class_weight import compute_class_weight


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

MANIFEST_PATH = Path("data/manifests/handpd_manifest.csv")
IMG_SIZE = 224
BATCH_SIZE = 16          # small dataset -- small batch size is fine and helps regularize
HEAD_EPOCHS = 20         # phase 1: frozen backbone, train head only
FINETUNE_EPOCHS = 30     # phase 2: unfreeze last block(s), fine-tune
HEAD_LR = 1e-4
FINETUNE_LR = 1e-5       # lower LR for fine-tuning -- protects pretrained weights
UNFREEZE_FRACTION = 0.25 # unfreeze last 25% of backbone layers in phase 2
OUTPUT_DIR = Path("outputs")
RANDOM_SEED = 42

LABEL_TO_INT = {"healthy": 0, "parkinson": 1}


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def load_manifest(fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} not found -- run data_pipeline.py first to "
            f"generate it."
        )
    df = pd.read_csv(MANIFEST_PATH)
    if fold not in df["fold"].unique():
        raise ValueError(f"Fold {fold} not found in manifest. Available folds: "
                          f"{sorted(df['fold'].unique())}")

    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)
    return train_df, val_df


def decode_and_preprocess(image_path, label, training: bool):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])

    if training:
        # Rotation + brightness/contrast jitter -- safe augmentations that
        # don't distort the tremor signal. Deliberately NOT flipping
        # (horizontal/vertical) since drawing direction may carry clinical
        # meaning and flipping could create unrealistic samples.
        img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)

    img = preprocess_input(img)  # EfficientNet-specific normalization
    return img, label


def build_dataset(df: pd.DataFrame, training: bool) -> tf.data.Dataset:
    paths = df["image_path"].values
    labels = df["label"].map(LABEL_TO_INT).values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=RANDOM_SEED)
    ds = ds.map(
        lambda p, l: decode_and_preprocess(p, l, training=training),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model() -> tuple[keras.Model, keras.Model]:
    """
    Returns (model, backbone) so the caller can toggle backbone.trainable
    between phase 1 (frozen) and phase 2 (partially unfrozen).
    """
    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg",
    )
    backbone.trainable = False  # frozen for phase 1

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = backbone(inputs, training=False)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="efficientnetb0_baseline")
    return model, backbone


def unfreeze_last_fraction(backbone: keras.Model, fraction: float):
    n_layers = len(backbone.layers)
    n_unfreeze = int(n_layers * fraction)
    backbone.trainable = True
    for layer in backbone.layers[: n_layers - n_unfreeze]:
        layer.trainable = False
    # Keep BatchNorm layers frozen even within the unfrozen portion --
    # standard practice, prevents BN statistics from being corrupted by
    # a small batch size / small dataset during fine-tuning.
    for layer in backbone.layers[n_layers - n_unfreeze:]:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    print(f"Unfroze last {n_unfreeze}/{n_layers} backbone layers "
          f"(BatchNorm layers kept frozen).")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def get_class_weights(train_df: pd.DataFrame) -> dict:
    y = train_df["label"].map(LABEL_TO_INT).values
    weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y)
    class_weights = {0: weights[0], 1: weights[1]}
    print(f"Class weights (healthy, parkinson): {class_weights}")
    return class_weights


def get_optimizer(lr: float):
    try:
        return keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
    except AttributeError:
        # Older TF versions: AdamW lives under experimental
        return keras.optimizers.experimental.AdamW(learning_rate=lr, weight_decay=1e-4)


def train(fold: int):
    OUTPUT_DIR.mkdir(exist_ok=True)
    checkpoint_path = OUTPUT_DIR / f"baseline_fold{fold}_best.keras"

    train_df, val_df = load_manifest(fold)
    print(f"Train: {len(train_df)} images ({train_df['subject_id'].nunique()} subjects) | "
          f"Val: {len(val_df)} images ({val_df['subject_id'].nunique()} subjects)")

    train_ds = build_dataset(train_df, training=True)
    val_ds = build_dataset(val_df, training=False)
    class_weights = get_class_weights(train_df)

    model, backbone = build_model()

    metrics = [
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="roc_auc", curve="ROC"),
        keras.metrics.AUC(name="pr_auc", curve="PR"),
    ]

    # --- Phase 1: train head only, backbone frozen ---
    print("\n=== Phase 1: training head (backbone frozen) ===")
    model.compile(optimizer=get_optimizer(HEAD_LR), loss="binary_crossentropy", metrics=metrics)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=HEAD_EPOCHS,
        class_weight=class_weights,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=6, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max", factor=0.5, patience=3),
        ],
    )

    # --- Phase 2: unfreeze last 25%, fine-tune at lower LR ---
    print("\n=== Phase 2: fine-tuning (last 25% of backbone unfrozen) ===")
    unfreeze_last_fraction(backbone, UNFREEZE_FRACTION)
    model.compile(optimizer=get_optimizer(FINETUNE_LR), loss="binary_crossentropy", metrics=metrics)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINETUNE_EPOCHS,
        class_weight=class_weights,
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
    print(f"\nFinal validation metrics (fold {fold}):")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")


def check_data(fold: int):
    """Sanity-check the pipeline without training: confirm images load,
    shapes are correct, and labels/class balance look right."""
    train_df, val_df = load_manifest(fold)
    print(f"Train: {len(train_df)} images, Val: {len(val_df)} images")
    print("\nTrain label balance:\n", train_df["label"].value_counts())
    print("\nVal label balance:\n", val_df["label"].value_counts())

    ds = build_dataset(train_df, training=True)
    for images, labels in ds.take(1):
        print(f"\nBatch image shape: {images.shape}, dtype: {images.dtype}")
        print(f"Batch label shape: {labels.shape}, sample labels: {labels.numpy()[:8]}")
        print(f"Pixel value range after preprocessing: "
              f"[{images.numpy().min():.2f}, {images.numpy().max():.2f}]")
    print("\nData pipeline check passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, help="Which fold to use as validation (0-4)")
    parser.add_argument("--check_data", action="store_true", help="Verify data loads correctly, skip training")
    args = parser.parse_args()

    if args.check_data:
        check_data(args.fold)
    else:
        train(args.fold)