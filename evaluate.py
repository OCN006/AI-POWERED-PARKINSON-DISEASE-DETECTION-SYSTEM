import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Imports the dataset logic and LSTM architecture directly from your main.py file
from main import BidslabSleepFusionDataset, DualSensorFusionModel


def evaluate_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating model on: {device}")

    dataset_path = os.path.join(
        "Sleepdata",
        "a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0"
    )

    # Loads the 30-second window dataset
    dataset = BidslabSleepFusionDataset(root_dir=dataset_path)

    if len(dataset) == 0:
        print("Error: No data found. Check your dataset path.")
        return

    # Recreates the EXACT 80/20 split used during training (Seed 42)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_ds = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    # Loads the saved LSTM model's "brain"
    model = DualSensorFusionModel(num_classes=6).to(device)

    try:
        model.load_state_dict(torch.load("bidslab_sleep_model.pth", map_location=device, weights_only=True))
        print("Trained model 'bidslab_sleep_model.pth' loaded successfully!\n")
    except FileNotFoundError:
        print("Error: 'bidslab_sleep_model.pth' not found. Run main.py to train it first.")
        return

    model.eval()
    all_preds = []
    all_targets = []

    print(f"Running predictions on the {len(test_ds)} test epochs...")
    with torch.no_grad():
        for hr_batch, motion_batch, labels in test_loader:
            hr_batch = hr_batch.to(device)
            motion_batch = motion_batch.to(device)

            outputs = model(hr_batch, motion_batch)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    # Calculate and print metrics
    acc = accuracy_score(all_targets, all_preds)

    print("\n" + "=" * 50)
    print("DETAILED TEST EVALUATION RESULTS")
    print("=" * 50)
    print(f"Overall Test Accuracy : {acc * 100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(all_targets, all_preds, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))


if __name__ == "__main__":
    evaluate_model()