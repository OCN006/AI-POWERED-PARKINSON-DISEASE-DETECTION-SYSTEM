import os
import torch
import pandas as pd
import numpy as np
from main import DualSensorFusionModel  # Imports the architecture from your main.py file

def predict_single_session():
    # 1. Set up your hardware device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on: {device}")

    # 2. Recreate the model structure and load the saved "brain" (weights)
    model = DualSensorFusionModel(num_classes=6).to(device)
    model.load_state_dict(torch.load("bidslab_sleep_model.pth", map_location=device, weights_only=True))
    model.eval()  # Set model to evaluation mode
    print("Model 'bidslab_sleep_model.pth' loaded successfully!")

    # 3. Point to a specific patient session folder to test
    # (Targeting subject Bidslab00, session 1 as an example)
    target_folder = os.path.join(
        "Sleepdata",
        "a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0",
        "Bidslab00",
        "1"
    )

    hr_file = os.path.join(target_folder, "hr.csv")
    motion_file = os.path.join(target_folder, "motion.csv")

    if not os.path.exists(hr_file) or not os.path.exists(motion_file):
        print(f"Error: Could not find files in {target_folder}")
        return

    # 4. Load and process Heart Rate data (matching the training pipeline)
    df_hr = pd.read_csv(hr_file)
    hr_vals = df_hr.iloc[:, 0].values.astype(np.float32)
    hr_vals = hr_vals[:1000] if len(hr_vals) > 1000 else np.pad(hr_vals, (0, 1000 - len(hr_vals)), 'constant')
    # Add batch and channel dimensions -> Shape: [1, 1, 1000]
    hr_tensor = torch.tensor(hr_vals).unsqueeze(0).unsqueeze(0).to(device)

    # 5. Load and process Motion data (matching the training pipeline)
    df_motion = pd.read_csv(motion_file)
    motion_vals = df_motion.iloc[:, 0].values.astype(np.float32)
    motion_vals = motion_vals[:1000] if len(motion_vals) > 1000 else np.pad(motion_vals, (0, 1000 - len(motion_vals)), 'constant')
    # Add batch and channel dimensions -> Shape: [1, 1, 1000]
    motion_tensor = torch.tensor(motion_vals).unsqueeze(0).unsqueeze(0).to(device)

    # 6. Pass data through the fused model to get a prediction
    with torch.no_grad():
        outputs = model(hr_tensor, motion_tensor)
        predicted_class = torch.argmax(outputs, dim=1).item()

    print("-" * 40)
    print(f"Patient Folder: Bidslab00 / Session 1")
    print(f"Predicted Sleep Stage Class: {predicted_class}")
    print("-" * 40)

if __name__ == "__main__":
    predict_single_session()