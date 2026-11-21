import argparse
import os

import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler


# ==========================================
# 1. EPOCH-BASED DATASET LOADER (Fixed Normalization)
# ==========================================
class BidslabSleepFusionDataset(Dataset):
    def __init__(self, root_dir, fixed_len=400, seq_len=5):
        self.hr_data = []
        self.motion_data = []
        self.labels = []
        self.subject_ids = []
        self.seq_len = seq_len

        print(f"Scanning directory and slicing data into clinical 30-second epochs...")

        for dirpath, dirnames, filenames in os.walk(root_dir):
            if "hr.csv" in filenames and "motion.csv" in filenames and "labels.mat" in filenames:

                labels_array = self._extract_labels_array(os.path.join(dirpath, "labels.mat"))
                num_epochs = len(labels_array)

                if num_epochs == 0:
                    continue

                hr_windows = self._load_slice_and_normalize(os.path.join(dirpath, "hr.csv"), num_epochs, fixed_len)
                motion_windows = self._load_slice_and_normalize(os.path.join(dirpath, "motion.csv"), num_epochs,
                                                                fixed_len)

                if hr_windows is not None and motion_windows is not None:
                    relative_dir = os.path.relpath(dirpath, root_dir)
                    subject_id = relative_dir.split(os.sep)[0]
                    for i in range(num_epochs):
                        if 0 <= labels_array[i] < 6:
                            self.hr_data.append(hr_windows[i])
                            self.motion_data.append(motion_windows[i])
                            self.labels.append(torch.tensor(labels_array[i], dtype=torch.long))
                            self.subject_ids.append(subject_id)

        print(f"Successfully loaded {len(self.labels)} individual sleep epochs into memory!")

    def __len__(self):
        return len(self.labels)

    def _extract_labels_array(self, mat_path):
        try:
            mat_data = scipy.io.loadmat(mat_path)
            for key in mat_data:
                if not key.startswith('__'):
                    return np.array(mat_data[key]).flatten().astype(int)
            return np.array([])
        except Exception:
            return np.array([])

    def _load_slice_and_normalize(self, csv_path, num_epochs, fixed_len):
        df = pd.read_csv(csv_path)
        values = df.iloc[:, 0].values.astype(np.float32)

        # CRITICAL FIX: Normalize the ENTIRE NIGHT first to preserve absolute highs/lows
        std = np.std(values)
        if std > 1e-6:
            values = (values - np.mean(values)) / std
        else:
            values = values - np.mean(values)

        # NOW slice into epochs
        chunk_size = len(values) // num_epochs
        if chunk_size == 0:
            return None

        windows = []
        for i in range(num_epochs):
            window = values[i * chunk_size: (i + 1) * chunk_size]

            if len(window) > fixed_len:
                window = window[:fixed_len]
            else:
                window = np.pad(window, (0, fixed_len - len(window)), 'constant')

            windows.append(torch.tensor(window).unsqueeze(0))

        return windows

    def __getitem__(self, idx):
        hr_seq = []
        motion_seq = []
        target_subject = self.subject_ids[idx]
        
        for i in range(idx - self.seq_len + 1, idx + 1):
            if i < 0 or self.subject_ids[i] != target_subject:
                hr_seq.append(torch.zeros_like(self.hr_data[idx]))
                motion_seq.append(torch.zeros_like(self.motion_data[idx]))
            else:
                hr_seq.append(self.hr_data[i])
                motion_seq.append(self.motion_data[i])
                
        hr_seq = torch.stack(hr_seq) 
        motion_seq = torch.stack(motion_seq)
        return hr_seq, motion_seq, self.labels[idx]


def create_subject_level_split(dataset, test_size=0.2, random_state=42):
    """Split epochs so that no subject contributes to both train and test sets."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_indices, test_indices = next(
        splitter.split(np.zeros(len(dataset)), dataset.labels, groups=dataset.subject_ids)
    )
    return Subset(dataset, train_indices.tolist()), Subset(dataset, test_indices.tolist())


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    return device


def print_evaluation_metrics(targets, predictions, probabilities):
    class_names = ["Wake", "N1", "N2", "N3", "REM", "Unscored"]
    accuracy = accuracy_score(targets, predictions)

    print("\n" + "=" * 50)
    print("FINAL EVALUATION METRICS")
    print("=" * 50)
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print("\nClassification report (precision, recall, F1-score):")
    print(classification_report(
        targets, predictions, labels=list(range(6)), target_names=class_names, zero_division=0
    ))
    print("Macro ROC-AUC (one-vs-rest):")
    try:
        print(f"{roc_auc_score(targets, probabilities, labels=list(range(6)), multi_class='ovr', average='macro'):.4f}")
    except ValueError as error:
        print(f"Unavailable: {error}")
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(targets, predictions, labels=list(range(6))))


# ==========================================
# 2. IMPROVED CRNN WITH ATTENTION POOLING
# ==========================================
class SensorEncoder(nn.Module):
    """Three-block 1D-CNN -> BiLSTM -> attention-weighted pooling."""

    def __init__(self, in_channels=1, out_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(25),
        )
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=32, num_layers=1,
            batch_first=True, bidirectional=True,
        )
        self.attn = nn.Linear(64, 1)
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.cnn(x)                    # [B, 64, 25]
        x = x.permute(0, 2, 1)             # [B, 25, 64]
        lstm_out, _ = self.lstm(x)          # [B, 25, 64]
        attn_w = torch.softmax(self.attn(lstm_out), dim=1)
        pooled = (lstm_out * attn_w).sum(dim=1)
        return torch.relu(self.fc(pooled))


class DualSensorFusionModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.hr_encoder = SensorEncoder(out_dim=64)
        self.motion_encoder = SensorEncoder(out_dim=64)

        self.context_gru = nn.GRU(
            input_size=128, hidden_size=64, num_layers=1, batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, hr_x, motion_x):
        B, S, C, L = hr_x.shape
        
        hr_x = hr_x.view(B * S, C, L)
        motion_x = motion_x.view(B * S, C, L)

        hr_features = self.hr_encoder(hr_x)
        motion_features = self.motion_encoder(motion_x)

        fused_features = torch.cat((hr_features, motion_features), dim=1)
        fused_features = fused_features.view(B, S, 128)
        
        lstm_out, _ = self.context_gru(fused_features)
        last_out = lstm_out[:, -1, :]
        
        logits = self.classifier(last_out)
        return logits


# ==========================================
# 3. FOCAL LOSS
# ==========================================
class FocalLoss(nn.Module):
    """Focal loss with optional per-class weights and label smoothing."""

    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets, weight=self.weight,
            reduction="none", label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


# ==========================================
# 4. CLI
# ==========================================
def parse_training_args():
    p = argparse.ArgumentParser(description="Train the six-class sleep staging model.")
    p.add_argument("--loss-type", choices=["weighted_ce", "focal"], default="focal")
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--no-balanced-sampling", action="store_true")
    p.add_argument("--unweighted", action="store_true", help="Disable class weights to maximize raw accuracy.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seq-len", type=int, default=21, help="Number of epochs in the sliding window context.")
    p.add_argument("--save-path", default="bidslab_sleep_model.pth")
    return p.parse_args()


# ==========================================
# 5. TRAINING ENGINE
# ==========================================
def train_and_evaluate():
    args = parse_training_args()
    class_names = ["Wake", "N1", "N2", "N3", "REM", "Unscored"]

    dataset_path = os.path.join(
        "Sleepdata",
        "a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0",
    )
    dataset = BidslabSleepFusionDataset(root_dir=dataset_path, seq_len=args.seq_len)
    if len(dataset) == 0:
        print("\nCRITICAL ERROR: No data found.")
        return

    train_ds, test_ds = create_subject_level_split(dataset)
    train_labels = torch.tensor(
        [dataset.labels[idx].item() for idx in train_ds.indices], dtype=torch.long,
    )
    class_counts = torch.bincount(train_labels, minlength=6).float()
    total = float(class_counts.sum())

    print("\n--- Training class distribution ---")
    for i, name in enumerate(class_names):
        n = int(class_counts[i].item())
        print(f"  {name:>10}: {n:>7}  ({n / total * 100:5.1f}%)")

    # --- Balanced sampling (only balances batch composition) ---
    if not args.no_balanced_sampling:
        sw = 1.0 / class_counts[train_labels]
        sampler = WeightedRandomSampler(sw, num_samples=len(train_ds), replacement=True,
                                        generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
        print("Balanced sampling: ENABLED")
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        print("Balanced sampling: DISABLED")

    test_loader = (
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        if len(test_ds) > 0 else None
    )

    device = get_device()
    model = DualSensorFusionModel(num_classes=6).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # --- Sqrt-inverse class weights, clipped (mild, since sampling already balances) ---
    if args.unweighted:
        cw = torch.ones(6).to(device)
        print("Class weights: UNWEIGHTED (All 1.0) - Accuracy Maximizer Mode")
    else:
        cw = torch.sqrt(total / (6.0 * class_counts)).clamp(0.8, 2.5).to(device)
        print("Class weights (sqrt-inverse, clipped [0.8, 2.5]):")
        for i, name in enumerate(class_names):
            print(f"  {name:>10}: {cw[i].item():.4f}")

    # --- Loss ---
    if args.loss_type == "focal":
        criterion = FocalLoss(weight=cw, gamma=args.gamma, label_smoothing=args.label_smoothing).to(device)
        print(f"Loss: Focal (gamma={args.gamma}, smoothing={args.label_smoothing})")
    else:
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)
        print(f"Loss: Weighted CE (smoothing={args.label_smoothing})")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_macro_f1 = 0.0
    print(f"\nTraining {args.epochs} epochs on {device}")
    print("-" * 100)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for hr_batch, motion_batch, labels in train_loader:
            hr_batch = hr_batch.to(device)
            motion_batch = motion_batch.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(hr_batch, motion_batch)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        lr_now = scheduler.get_last_lr()[0]

        # --- Per-class F1 validation ---
        val_summary = ""
        if test_loader is not None:
            model.eval()
            v_preds, v_true = [], []
            with torch.no_grad():
                for hr_b, mot_b, lbl in test_loader:
                    out = model(hr_b.to(device), mot_b.to(device))
                    v_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                    v_true.extend(lbl.numpy())
            pcf1 = f1_score(v_true, v_preds, labels=list(range(6)), average=None, zero_division=0)
            macro = float(np.mean(pcf1))
            acc = accuracy_score(v_true, v_preds)
            parts = " ".join(f"{class_names[i][:4]}={pcf1[i]:.3f}" for i in range(6))
            saved_tag = ""
            if macro > best_macro_f1:
                best_macro_f1 = macro
                torch.save(model.state_dict(), args.save_path)
                saved_tag = " *BEST*"
            val_summary = f"  | acc={acc:.3f} macF1={macro:.3f} [{parts}]{saved_tag}"

        print(f"Ep [{epoch:02d}/{args.epochs:02d}] loss={avg_loss:.4f} lr={lr_now:.6f}{val_summary}")

    # --- Reload best checkpoint and final eval ---
    print(f"\nBest macro F1: {best_macro_f1:.4f}")
    model.load_state_dict(torch.load(args.save_path, map_location=device, weights_only=True))

    if test_loader is not None:
        model.eval()
        all_preds, all_targets, all_probs = [], [], []
        with torch.no_grad():
            for hr_batch, motion_batch, labels in test_loader:
                outputs = model(hr_batch.to(device), motion_batch.to(device))
                all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                all_targets.extend(labels.numpy())
                all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
        print_evaluation_metrics(all_targets, all_preds, np.asarray(all_probs))

    print(f"\nBest model saved to: {args.save_path}")


if __name__ == "__main__":
    train_and_evaluate()