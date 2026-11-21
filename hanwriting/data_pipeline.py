"""
Data pipeline for Parkinson's handwriting detection project.

Builds unified manifests (CSV) for HandPD and NewHandPD datasets, with
SUBJECT-LEVEL stratified 5-fold splits on HandPD (train/val) and NewHandPD
kept entirely separate as an external test set.

WHY SUBJECT-LEVEL SPLITTING MATTERS:
Both datasets have multiple images per person (e.g. 4 spirals per subject
in HandPD). If you split by image instead of by subject, the same person's
handwriting can appear in both train and validation folds, letting the
model partially memorize a person's style rather than learn the disease
signal. This inflates CV accuracy and won't hold up under review.

ACTUAL DATA STRUCTURE (verified from the real downloaded zips):
1. HandPD: class comes from the FOLDER name, not a CSV or the filename.
   Spiral_HandPD.zip -> SpiralControl/ (healthy) and SpiralPatients/
   (parkinson) subfolders. Meander_HandPD.zip is assumed to follow the
   same pattern (MeanderControl/ MeanderPatients/) -- verify this once
   you extract it, folder names may differ slightly.
   Filenames inside: {4-digit subject id}-{image number 1-4}.jpg
   e.g. 0068-1.jpg, 0068-2.jpg ... 0068-4.jpg. Each folder also contains
   a harmless .DS_Store file (macOS metadata) which is skipped.
2. NewHandPD: filenames follow the documented pattern spN-H{id}.jpg /
   spN-P{id}.jpg (spiral) and meN-H{id}.jpg / meN-P{id}.jpg (meander) --
   VERIFY this against your actual unzipped filenames before trusting the
   regex below; adjust NEWHANDPD_PATTERN if the real filenames differ.

Run `python data_pipeline.py --check` first to print a sample of parsed
filenames/labels for manual verification before doing the full build.
"""

import argparse
import csv
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# ---------------------------------------------------------------------------
# CONFIG -- edit these paths to match where you extracted the zip files
# ---------------------------------------------------------------------------

HANDPD_SPIRAL_CONTROL_DIR = Path("data/raw/handpd/Spiral_HandPD/SpiralControl")
HANDPD_SPIRAL_PATIENT_DIR = Path("data/raw/handpd/Spiral_HandPD/SpiralPatients")
HANDPD_MEANDER_CONTROL_DIR = Path("data/raw/handpd/Meander_HandPD/MeanderControl")
HANDPD_MEANDER_PATIENT_DIR = Path("data/raw/handpd/Meander_HandPD/MeanderPatients")

# Filename pattern inside SpiralControl/SpiralPatients: 0068-1.jpg
# Group 1 = subject id, group 2 = image number (not needed for grouping,
# but kept in case you want per-image tracking later)
HANDPD_FILENAME_PATTERN = re.compile(r"^(\d+)-(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)

NEWHANDPD_HEALTHY_SPIRAL_DIR = Path("data/raw/newhandpd/HealthySpiral")
NEWHANDPD_PATIENT_SPIRAL_DIR = Path("data/raw/newhandpd/PatientSpiral")
NEWHANDPD_HEALTHY_MEANDER_DIR = Path("data/raw/newhandpd/HealthyMeander")
NEWHANDPD_PATIENT_MEANDER_DIR = Path("data/raw/newhandpd/PatientMeander")

OUTPUT_DIR = Path("data/manifests")
N_FOLDS = 5
RANDOM_SEED = 42

# NewHandPD filename pattern: e.g. sp1-H12.jpg / sp1-P7.jpg / me2-H3.jpg
# Group 1 = class letter (H/P), Group 2 = subject id number
NEWHANDPD_PATTERN = re.compile(r"[a-zA-Z]+\d*-([HP])(\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# HandPD: build manifest from folder name (class) + filename (subject id)
# ---------------------------------------------------------------------------

def build_handpd_manifest(folder: Path, label: str, task: str) -> pd.DataFrame:
    """
    folder: e.g. .../SpiralControl or .../SpiralPatients
    label: 'healthy' or 'parkinson' -- determined by which folder this is,
           since HandPD splits classes into separate folders rather than
           encoding class in the filename or a CSV.
    task: 'spiral' or 'meander' -- carried through to the manifest.
    """
    if not folder.exists():
        raise FileNotFoundError(
            f"Expected folder {folder} -- update the HANDPD_* paths above "
            f"to match where you extracted the HandPD zip files. Note the "
            f"real structure is SpiralControl/ + SpiralPatients/ subfolders, "
            f"not a flat folder + CSV."
        )

    rows = []
    skipped = []
    for img_path in sorted(folder.glob("*.*")):
        if img_path.name.startswith(".") or img_path.name == "Thumbs.db":
            continue  # skip .DS_Store and other OS junk files
        m = HANDPD_FILENAME_PATTERN.match(img_path.name)
        if not m:
            skipped.append(img_path.name)
            continue
        subject_num = m.group(1)
        rows.append({
            "image_path": str(img_path),
            "subject_id": f"handpd_{subject_num}",
            "label": label,
            "dataset": "handpd",
            "task": task,
        })

    if skipped:
        print(f"  [warn] {len(skipped)} files in {folder} did not match the "
              f"expected NNNN-N.jpg pattern and were skipped: {skipped[:5]}"
              f"{' ...' if len(skipped) > 5 else ''}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# NewHandPD: build manifest by parsing filenames (class is already split
# into separate healthy/patient folders, so this mostly just extracts
# subject_id for grouping)
# ---------------------------------------------------------------------------

def build_newhandpd_manifest(folder: Path, label: str, task: str) -> pd.DataFrame:
    if not folder.exists():
        raise FileNotFoundError(
            f"Expected folder {folder} -- update the NEWHANDPD_* paths above "
            f"to match where you extracted the NewHandPD zip files."
        )

    rows = []
    unparsed = 0
    for img_path in sorted(folder.glob("*.*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        m = NEWHANDPD_PATTERN.search(img_path.stem)
        if not m:
            unparsed += 1
            continue
        subject_num = m.group(2)
        rows.append({
            "image_path": str(img_path),
            "subject_id": f"newhandpd_{label}_{subject_num}",
            "label": "parkinson" if label == "P" else "healthy",
            "dataset": "newhandpd",
            "task": task,
        })

    if unparsed:
        print(f"  [warn] {unparsed} filenames in {folder} did not match "
              f"the expected pattern -- inspect them manually and fix "
              f"NEWHANDPD_PATTERN if needed.")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Subject-level stratified K-fold (HandPD only -- NewHandPD stays untouched
# as external test set)
# ---------------------------------------------------------------------------

def assign_folds(df: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    """
    Adds a 'fold' column (0..n_folds-1) using StratifiedGroupKFold, grouped
    by subject_id so no subject's images cross fold boundaries.
    """
    df = df.reset_index(drop=True).copy()
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    y = df["label"].values
    groups = df["subject_id"].values

    df["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(sgkf.split(df, y, groups)):
        df.loc[val_idx, "fold"] = fold_idx

    # Sanity check: verify no subject appears in more than one fold
    leak_check = df.groupby("subject_id")["fold"].nunique()
    leaked = leak_check[leak_check > 1]
    if len(leaked):
        raise RuntimeError(
            f"Subject leakage detected across folds for subjects: "
            f"{list(leaked.index)}. This should not happen with "
            f"StratifiedGroupKFold -- check for duplicate subject_id "
            f"formatting issues."
        )

    return df


def print_fold_summary(df: pd.DataFrame, name: str):
    print(f"\n{name} fold summary:")
    summary = df.groupby(["fold", "label"]).size().unstack(fill_value=0)
    print(summary)
    n_subjects = df["subject_id"].nunique()
    print(f"  total images: {len(df)}, unique subjects: {n_subjects}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(check_only: bool = False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building HandPD manifest (spiral)...")
    handpd_spiral = pd.concat([
        build_handpd_manifest(HANDPD_SPIRAL_CONTROL_DIR, "healthy", "spiral"),
        build_handpd_manifest(HANDPD_SPIRAL_PATIENT_DIR, "parkinson", "spiral"),
    ], ignore_index=True)
    print("Building HandPD manifest (meander)...")
    handpd_meander = pd.concat([
        build_handpd_manifest(HANDPD_MEANDER_CONTROL_DIR, "healthy", "meander"),
        build_handpd_manifest(HANDPD_MEANDER_PATIENT_DIR, "parkinson", "meander"),
    ], ignore_index=True)
    handpd = pd.concat([handpd_spiral, handpd_meander], ignore_index=True)

    print("Building NewHandPD manifest...")
    newhandpd = pd.concat([
        build_newhandpd_manifest(NEWHANDPD_HEALTHY_SPIRAL_DIR, "H", "spiral"),
        build_newhandpd_manifest(NEWHANDPD_PATIENT_SPIRAL_DIR, "P", "spiral"),
        build_newhandpd_manifest(NEWHANDPD_HEALTHY_MEANDER_DIR, "H", "meander"),
        build_newhandpd_manifest(NEWHANDPD_PATIENT_MEANDER_DIR, "P", "meander"),
    ], ignore_index=True)

    if check_only:
        print("\n--- CHECK MODE: sample of parsed rows, verify before full run ---")
        print("\nHandPD sample:")
        print(handpd.sample(min(10, len(handpd)), random_state=0).to_string(index=False))
        print("\nNewHandPD sample:")
        print(newhandpd.sample(min(10, len(newhandpd)), random_state=0).to_string(index=False))
        print(f"\nHandPD total: {len(handpd)} images, {handpd['subject_id'].nunique()} subjects")
        print(f"NewHandPD total: {len(newhandpd)} images, {newhandpd['subject_id'].nunique()} subjects")
        print("\nLabel balance:")
        print("HandPD:\n", handpd["label"].value_counts())
        print("NewHandPD:\n", newhandpd["label"].value_counts())
        return

    if len(handpd) == 0:
        raise RuntimeError("HandPD manifest is empty -- check paths/CSV before proceeding.")
    if len(newhandpd) == 0:
        raise RuntimeError("NewHandPD manifest is empty -- check paths before proceeding.")

    print("\nAssigning subject-stratified folds to HandPD...")
    handpd = assign_folds(handpd, N_FOLDS, RANDOM_SEED)
    print_fold_summary(handpd, "HandPD")

    # NewHandPD gets no fold assignment -- it's the external test set,
    # used once at the very end, never during training or tuning.
    newhandpd["fold"] = "external_test"

    handpd_out = OUTPUT_DIR / "handpd_manifest.csv"
    newhandpd_out = OUTPUT_DIR / "newhandpd_manifest.csv"
    handpd.to_csv(handpd_out, index=False)
    newhandpd.to_csv(newhandpd_out, index=False)

    print(f"\nSaved: {handpd_out}")
    print(f"Saved: {newhandpd_out}")
    print("\nNext step: build a tf.data / torch Dataset that reads these "
          "manifests, filters by fold for train/val, and applies the "
          "resize + normalization + augmentation pipeline.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                         help="Print a sample of parsed data without building folds -- run this first.")
    args = parser.parse_args()
    main(check_only=args.check)