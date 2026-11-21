"""
Test the model using ALL 5 folds together (ensemble) instead of picking
just one -- generally more reliable than trusting a single fold's
checkpoint, since it averages out any one fold's quirks.

Usage:
    python predict_ensemble.py --model_type baseline --image path/to/image.jpg
    python predict_ensemble.py --model_type cbam --image path/to/image.jpg
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input

from train_baseline import OUTPUT_DIR

IMG_SIZE = 224
N_FOLDS = 5

# --- EDIT THESE TWO LINES TO TEST A DIFFERENT IMAGE OR MODEL ---
IMAGE_PATH = "data\images.jpeg"
MODEL_TYPE = "cbam"  # "baseline" or "cbam"
# -----------------------------------------------------------


def load_image(image_path: str):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = preprocess_input(img)
    return tf.expand_dims(img, axis=0)


def main(model_type: str, image_path: str):
    img = load_image(image_path)
    fold_probs = []

    for fold in range(N_FOLDS):
        checkpoint = OUTPUT_DIR / f"{model_type}_fold{fold}_best.keras"
        print(f"Loading fold {fold}: {checkpoint}")
        model = keras.models.load_model(checkpoint, safe_mode=False)
        prob = model.predict(img, verbose=0)[0, 0]
        fold_probs.append(prob)
        print(f"  fold {fold} says parkinson probability: {prob:.4f}")

    avg_prob = np.mean(fold_probs)
    label = "PARKINSON" if avg_prob >= 0.5 else "HEALTHY"
    confidence = avg_prob if avg_prob >= 0.5 else 1 - avg_prob

    print(f"\n=== Ensemble result ({model_type}, {N_FOLDS} folds averaged) ===")
    print(f"Image: {image_path}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence * 100:.1f}%")
    print(f"Individual fold votes: {[f'{p:.3f}' for p in fold_probs]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["baseline", "cbam"], default=MODEL_TYPE,
                         help=f"Defaults to the MODEL_TYPE global variable ('{MODEL_TYPE}')")
    parser.add_argument("--image", default=IMAGE_PATH,
                         help=f"Defaults to the IMAGE_PATH global variable ('{IMAGE_PATH}')")
    args = parser.parse_args()
    main(args.model_type, args.image)