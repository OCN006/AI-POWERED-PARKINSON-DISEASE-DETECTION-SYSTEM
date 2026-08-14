# Dual-Sensor Sleep Staging CRNN

An advanced Deep Learning pipeline that predicts clinical sleep stages (Wake, Light Sleep, Deep Sleep, REM) using instantaneous heart rate and accelerometry data. 

This project tackles the complexities of real-world biological time-series data, transitioning from a basic CNN to a Convolutional Recurrent Neural Network (CRNN) to capture temporal physiological trends.

## 🧠 Architecture: Feature-Level Fusion
The core engine is a `DualSensorFusionModel` built in PyTorch that fuses two independent data streams:
1. **CNN Feature Extractor:** Two parallel 1D-Convolutional networks scan 30-second windows of Heart Rate and Motion data to extract micro-patterns (e.g., sudden spikes, stable rhythms).
2. **Bi-Directional LSTM:** A recurrent layer acts as the model's memory, tracking how physiological features evolve over time to differentiate between transition states.
3. **Fusion & Classification:** The temporal features from both sensors are concatenated and passed through a densely connected classifier.

## 🛠️ Engineering Challenges Solved
Working with the [BIDSleep Dataset](https://physionet.org/content/bidsleep/1.0.0/) presented several clinical-grade machine learning challenges:

* **Epoch Slicing:** Translated the raw dataset from "whole-night" bulk labels into tens of thousands of clinically accurate 30-second temporal windows.
* **Global Z-Score Normalization:** Handled extreme value disparities between heart rate and motion. Applied *global* normalization across the entire night before slicing to ensure the absolute highs (Wake) and lows (Deep Sleep) were preserved for the neural network.
* **Extreme Class Imbalance:** Biological data is heavily skewed toward Light Sleep (N2). Implemented **Capped Class Weights (Max 5.0x)** inside the CrossEntropyLoss function to mathematically penalize the network for ignoring rare minority classes (like Deep Sleep and Wake) without destabilizing the LSTM gradients.

## 📊 Evaluation & Biological Limits
The model was evaluated on a test set of **43,688 individual sleep epochs**. 
It successfully identified Light Sleep, Deep Sleep, and Wake states with high precision. Misclassifications primarily occurred between REM sleep and Light Sleep—a known biological limit of wearable sensors, as REM ("paradoxical sleep") perfectly mimics waking heart rates and requires EEG (brainwave) data for perfect separation.

## 🚀 How to Run
1. Clone the repository.
2. Ensure you have the BIDSleep dataset downloaded and extracted.
3. Run the training pipeline:
   ```bash
   python main.py
