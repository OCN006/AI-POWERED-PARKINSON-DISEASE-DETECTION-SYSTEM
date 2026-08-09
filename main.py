import os
import pandas as pd
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight


# ==========================================
# 1. EPOCH-BASED DATASET LOADER (Fixed Normalization)
# ==========================================
class BidslabSleepFusionDataset(Dataset):
    def __init__(self, root_dir, fixed_len=400):
        self.hr_data = []
        self.motion_data = []
        self.labels = []

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
                    for i in range(num_epochs):
                        if 0 <= labels_array[i] < 6:
                            self.hr_data.append(hr_windows[i])
                            self.motion_data.append(motion_windows[i])
                            self.labels.append(torch.tensor(labels_array[i], dtype=torch.long))

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
        return self.hr_data[idx], self.motion_data[idx], self.labels[idx]


# ==========================================
# 2. CRNN: CONVOLUTIONAL RECURRENT NETWORK
# ==========================================
class SensorEncoder(nn.Module):
    def __init__(self, in_channels=1, out_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)
        out = torch.mean(lstm_out, dim=1)
        return torch.relu(self.fc(out))


class DualSensorFusionModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.hr_encoder = SensorEncoder(out_dim=64)
        self.motion_encoder = SensorEncoder(out_dim=64)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, hr_x, motion_x):
        hr_features = self.hr_encoder(hr_x)
        motion_features = self.motion_encoder(motion_x)

        fused_features = torch.cat((hr_features, motion_features), dim=1)
        logits = self.classifier(fused_features)
        return logits


# ==========================================
# 3. TRAINING ENGINE WITH CAPPED WEIGHTS
# ==========================================
def train_and_evaluate():
    dataset_path = os.path.join(
        "Sleepdata",
        "a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0"
    )

    dataset = BidslabSleepFusionDataset(root_dir=dataset_path)

    if len(dataset) == 0:
        print("\nCRITICAL ERROR: No data found.")
        return

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False) if test_size > 0 else None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualSensorFusionModel(num_classes=6).to(device)

    print("Calculating capped class weights...")
    train_labels = [train_ds.dataset[idx][2].item() for idx in train_ds.indices]
    unique_labels = np.unique(train_labels)
    computed_weights = compute_class_weight('balanced', classes=unique_labels, y=train_labels)

    # CRITICAL FIX: Cap the weights at 5.0 so the LSTM doesn't destabilize
    computed_weights = np.clip(computed_weights, 1.0, 5.0)

    full_weights = torch.ones(6, dtype=torch.float32)
    for cls_idx, weight in zip(unique_labels, computed_weights):
        if int(cls_idx) < 6:
            full_weights[int(cls_idx)] = float(weight)
    full_weights = full_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=full_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 15  # Increased slightly to give the LSTM time to learn
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\nStarting Fixed LSTM Training ({epochs} Epochs) on: {device}")
    print("-" * 50)

    for epoch in range(1, epochs + 1):
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
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} (LR: {current_lr:.6f})")

    if test_loader is None:
        return

    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for hr_batch, motion_batch, labels in test_loader:
            outputs = model(hr_batch.to(device), motion_batch.to(device))
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    print("\n" + "=" * 50)
    print("FINAL EVALUATION METRICS")
    print("=" * 50)
    acc = accuracy_score(all_targets, all_preds)
    print(f"Accuracy : {acc * 100:.2f}%")

    save_path = "bidslab_sleep_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel successfully saved to: {save_path}")


if __name__ == "__main__":
    train_and_evaluate()