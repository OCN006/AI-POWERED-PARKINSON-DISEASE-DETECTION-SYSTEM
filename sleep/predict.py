import os
import torch
import pandas as pd
import numpy as np
from main import DualSensorFusionModel

# Dictionary to translate the AI's math (0-5) into human medical terms
SLEEP_STAGES = {
    0: "Wake",
    1: "N1 (Drowsy Transition)",
    2: "N2 (Light Sleep)",
    3: "N3 (Deep Sleep)",
    4: "REM (Dreaming)",
    5: "Unscored / Sensor Error"
}


def load_and_preprocess_raw_data(csv_path, window_size=400):
    """Loads raw CSV data, normalizes it globally, and slices it into 30-second windows."""
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    values = df.iloc[:, 0].values.astype(np.float32)

    # 1. Global Normalization (Preserves absolute highs and lows)
    std = np.std(values)
    if std > 1e-6:
        values = (values - np.mean(values)) / std
    else:
        values = values - np.mean(values)

    # 2. Slice into 30-second (400 datapoint) windows
    num_windows = len(values) // window_size
    windows = []

    for i in range(num_windows):
        window = values[i * window_size: (i + 1) * window_size]
        windows.append(window)

    # Convert to PyTorch tensor format: [Batch, Channels, Length]
    tensor_data = torch.tensor(np.array(windows)).unsqueeze(1)
    return tensor_data


def create_sliding_windows(tensor_data, seq_len=5):
    num_epochs, C, L = tensor_data.shape
    windows = []
    for i in range(num_epochs):
        seq = []
        for j in range(i - seq_len + 1, i + 1):
            if j < 0:
                seq.append(torch.zeros_like(tensor_data[0]))
            else:
                seq.append(tensor_data[j])
        windows.append(torch.stack(seq))
    return torch.stack(windows)


def generate_hypnogram(hr_path, motion_path, model_path="bidslab_sleep_model.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading raw patient data...")
    hr_tensor = load_and_preprocess_raw_data(hr_path)
    motion_tensor = load_and_preprocess_raw_data(motion_path)

    if hr_tensor is None or motion_tensor is None:
        return

    # Ensure both sensors have the exact same number of 30-second windows
    min_length = min(len(hr_tensor), len(motion_tensor))
    hr_tensor = hr_tensor[:min_length]
    motion_tensor = motion_tensor[:min_length]
    
    # Create sequence context windows
    hr_tensor = create_sliding_windows(hr_tensor, seq_len=21).to(device)
    motion_tensor = create_sliding_windows(motion_tensor, seq_len=21).to(device)

    print(f"Processed {min_length} epochs (approx {min_length * 30 / 60:.1f} minutes of sleep).")

    # Load the trained AI brain
    model = DualSensorFusionModel(num_classes=6).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Please train the model first.")
        return

    model.eval()

    print("\n" + "=" * 50)
    print("AI SLEEP PREDICTION TIMELINE (HYPNOGRAM)")
    print("=" * 50)

    # Run Inference
    with torch.no_grad():
        outputs = model(hr_tensor, motion_tensor)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()

    # Print the timeline for the first 20 epochs (10 minutes) so it doesn't spam the console
    for minute_idx, pred in enumerate(predictions[:20]):
        time_elapsed = (minute_idx * 30)
        minutes = time_elapsed // 60
        seconds = time_elapsed % 60

        stage_name = SLEEP_STAGES.get(pred, "Unknown")
        print(f"Time {minutes:02d}m:{seconds:02d}s  -->  {stage_name}")

    if len(predictions) > 20:
        print(f"... and {len(predictions) - 20} more epochs predicted.")

    # Optional: Save full predictions to a file
    output_df = pd.DataFrame({"Epoch": range(len(predictions)), "Predicted_Stage_Code": predictions})
    output_df["Stage_Name"] = output_df["Predicted_Stage_Code"].map(SLEEP_STAGES)
    output_df.to_csv("predicted_sleep_timeline.csv", index=False)
    print("\nFull timeline saved to: predicted_sleep_timeline.csv")


if __name__ == "__main__":
    # =========================================================================
    # INSTRUCTIONS:
    # Grab ONE specific patient's folder from your BIDSleep dataset and paste
    # the paths to their hr.csv and motion.csv below to see their sleep!
    # =========================================================================

    PATIENT_HR_FILE = r"Sleepdata\a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0\1001\hr.csv"
    PATIENT_MOTION_FILE = r"Sleepdata\a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0\1001\motion.csv"

    generate_hypnogram(PATIENT_HR_FILE, PATIENT_MOTION_FILE)