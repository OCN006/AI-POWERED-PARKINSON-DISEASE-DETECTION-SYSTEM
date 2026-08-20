import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from main import BidslabSleepFusionDataset, DualSensorFusionModel, get_device, print_evaluation_metrics


def evaluate_new_dataset(new_data_folder):
    device = get_device()
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
    all_probabilities = []

    print(f"Running predictions on {len(external_dataset)} new unseen epochs...")
    with torch.no_grad():
        for hr_batch, motion_batch, labels in test_loader:
            hr_batch = hr_batch.to(device)
            motion_batch = motion_batch.to(device)

            outputs = model(hr_batch, motion_batch)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_probabilities.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    print("\n" + "=" * 50)
    print("EXTERNAL DATASET EVALUATION RESULTS")
    print("=" * 50)
    print_evaluation_metrics(all_targets, all_preds, np.asarray(all_probabilities))


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