import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from main import (
    BidslabSleepFusionDataset,
    DualSensorFusionModel,
    create_subject_level_split,
    get_device,
    print_evaluation_metrics,
)



def evaluate_model():
    device = get_device()

    dataset_path = os.path.join(
        "Sleepdata",
        "a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0"
    )

    # Loads the 30-second window dataset
    dataset = BidslabSleepFusionDataset(root_dir=dataset_path)

    if len(dataset) == 0:
        print("Error: No data found. Check your dataset path.")
        return

    # Recreates the exact subject-level 80/20 split used during training.
    _, test_ds = create_subject_level_split(dataset)

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
    all_probabilities = []

    print(f"Running predictions on the {len(test_ds)} test epochs...")
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
    print("DETAILED TEST EVALUATION RESULTS")
    print("=" * 50)
    print_evaluation_metrics(all_targets, all_preds, np.asarray(all_probabilities))


if __name__ == "__main__":
    evaluate_model()