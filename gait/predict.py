import os
import joblib
import numpy as np
from features import extract_window_features

DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "gait_pd_model.pkl")

def predict_gait_proba(filepath, model_path=DEFAULT_MODEL):
    """Returns Parkinson's probability (0.0 to 1.0) for a PhysioNet walking .txt file."""
    model = joblib.load(model_path)
    raw = np.loadtxt(filepath)
    if len(raw) < 1600:
        raise ValueError("Recording too short (requires at least 16 seconds at 100 Hz).")
        
    sensors = raw[500:-500, 1:19]
    scale = np.mean(sensors[:, 16] + sensors[:, 17]) + 1e-8
    sensors = sensors / scale
    
    windows = [extract_window_features(sensors[s : s + 500]) for s in range(0, len(sensors) - 500, 250)]
    X_new = np.nan_to_num(np.array(windows, dtype=np.float32))
    
    probs = model.predict_proba(X_new)[:, 1]
    return float(np.mean(probs))
