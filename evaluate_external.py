import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Imports your dataset loader and model architecture from main.py
from main import BidslabSleepFusionDataset, DualSensorFusionModel


def evaluate_new_dataset(new_data_folder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating model on: {device}")
    print(f"Loading external dataset from: {new_data_folder}")
    print("-" * 50)

    # Loads 100% of the data in the provided folder as test data
    external_dataset = BidslabSleepFusionDataset(root_dir=new_data_folder)

    if len(external_dataset) == 0:
        print("Error: No valid hr.csv, motion.csv, or labels.mat files found in this folder.")
        return

    # Use a large batch size for fast inference
    test_loader = DataLoader(external_dataset, batch_size=128, shuffle=False)

    # Load your trained LSTM model's "brain"
    model = DualSensorFusionModel(num_classes=6).to(device)
    model.load_state_dict(torch.load("bidslab_sleep_model.pth", map_location=device, weights_only=True))
    model.eval()
    print("Trained model 'bidslab_sleep_model.pth' loaded successfully!\n")

    all_preds = []
    all_targets = []

    print(f"Running predictions on {len(external_dataset)} new unseen epochs...")
    with torch.no_grad():
        for hr_batch, motion_batch, labels in test_loader:
            hr_batch = hr_batch.to(device)
            motion_batch = motion_batch.to(device)

            outputs = model(hr_batch, motion_batch)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    # Calculate final metrics
    acc = accuracy_score(all_targets, all_preds)

    print("\n" + "=" * 50)
    print("EXTERNAL DATASET EVALUATION RESULTS")
    print("=" * 50)
    print(f"Overall Accuracy : {acc * 100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(all_targets, all_preds, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))


if __name__ == "__main__":
    # =====================================================================
    # CHANGE THIS PATH to the folder containing your new patient data.
    # It will recursively search for all hr.csv and motion.csv files inside.
    # =====================================================================
    NEW_DATASET_PATH = r"D:\ParkinsonDisease\Path\To\Your\New\Dataset"

    if os.path.exists(NEW_DATASET_PATH):
        evaluate_new_dataset(NEW_DATASET_PATH)
    else:
        print(f"Error: The path '{NEW_DATASET_PATH}' does not exist. Please update the path.")