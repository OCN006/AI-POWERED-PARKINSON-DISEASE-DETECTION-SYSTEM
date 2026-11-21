# Parkinson's Disease Prediction: Voice Model

Voice-based module of an AI-based Parkinson's disease prediction system (multimodal project with separate sleep, handwriting and gait models). It classifies a person as Parkinson's (PD) or healthy from pre-extracted speech features and outputs a probability (0-1) that can be fused with the other modules.

> **Disclaimer:** This is an academic prototype for educational purposes. It is not a medical device and must not be used for diagnosis.

---

## Dataset

[UCI Parkinson's Disease Classification (id=470)](https://archive.ics.uci.edu/dataset/470/parkinson+s+disease+classification) (Sakar et al., 2019)

- 756 recordings from 252 subjects (188 PD, 64 healthy), 3 recordings per subject (sustained vowel /a/)
- 753 precomputed features: baseline (jitter, shimmer, HNR), time-frequency, MFCC, wavelet, vocal fold, and TQWT features
- The dataset is **not** included in this repository. Download `pd_speech_features.csv` from the link above and place it in the project root.

## Methodology

| Step | Detail |
|---|---|
| Splitting | **Subject-wise** (`StratifiedGroupKFold` on subject `id`), so recordings of the same person never appear in both train and test. A normal random split leaks identity and inflates accuracy. |
| Preprocessing | `StandardScaler`, then `VarianceThreshold`, then `SelectKBest` (ANOVA F-test), all inside a `Pipeline` so they are fit on training folds only |
| Models | SVM (RBF), Logistic Regression, Random Forest, XGBoost, soft-voting Ensemble (SVM + LR + XGBoost) |
| Imbalance | Class weights; evaluation with balanced accuracy, AUC and MCC alongside accuracy |
| Evaluation | 3 x repeated 5-fold subject-wise CV, plus a held-out set of ~20% of subjects (SVM tuned on the remaining subjects only) |
| Aggregation | The 3 recordings per person are averaged into **one prediction per subject** |
| Explainability | Random Forest feature importance and SHAP summary plot |

## Results

Subject-level, subject-wise CV (mean over 3 repeats):

| Model | Accuracy | Balanced Acc. | F1 | Sensitivity | Specificity | AUC | MCC |
|---|---|---|---|---|---|---|---|
| SVM | 0.807 | 0.711 | 0.875 | 0.906 | 0.516 | 0.807 | 0.458 |
| Logistic Regression | 0.772 | 0.748 | 0.840 | 0.798 | 0.698 | 0.833 | 0.460 |
| Random Forest | 0.817 | 0.690 | 0.886 | 0.949 | 0.432 | 0.844 | 0.467 |
| XGBoost | 0.831 | 0.754 | 0.889 | 0.910 | 0.599 | 0.861 | 0.535 |
| **Ensemble (SVM+LR+XGB)** | 0.824 | **0.757** | 0.883 | 0.894 | 0.620 | 0.847 | 0.526 |

Hold-out subjects (51 subjects, tuned models):

| Model | Accuracy | Balanced Acc. | AUC | MCC |
|---|---|---|---|---|
| SVM | 0.804 | 0.691 | 0.844 | 0.437 |
| Ensemble | 0.824 | 0.806 | 0.854 | 0.574 |

**Notes**
- Accuracy is about 82% under subject-independent validation, which is the realistic range for this dataset. Results far above this usually indicate data leakage.
- The hold-out contains only 51 subjects (13 healthy), so its numbers are noisy. The CV table is the main result.
- Specificity is lower than sensitivity because the dataset has about 3x more PD than healthy subjects.

## Installation

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
```

## Usage

1. Place `pd_speech_features.csv` in the project root.
2. Run the full pipeline:

```bash
python parkinson_voice_model.py
```

This runs model comparison, tuning, hold-out evaluation, plots, and saves everything to `voice_outputs/`:

| File | Description |
|---|---|
| `cv_results.csv`, `holdout_results.csv` | Result tables |
| `roc_confusion.png` | ROC curve and confusion matrix |
| `feature_importance.png`, `shap_summary.png`, `top_features.csv` | Explainability |
| `voice_model.pkl` | Final ensemble trained on all subjects, with feature column names |

### Getting a probability (for multimodal fusion)

```python
import pandas as pd
from parkinson_voice_model import predict_voice  # or copy the function

df = pd.read_csv("pd_speech_features.csv", skiprows=1)
rows = df[df["id"] == df["id"].iloc[0]]      # all recordings of one person
print(predict_voice(rows))                    # probability of Parkinson's (0-1)
```

Note: importing the module re-runs the pipeline. To avoid this, copy `predict_voice` into its own file.

## Limitations

- The model expects the 753 **precomputed UCI features**. Many (especially TQWT and wavelet features) cannot be reproduced exactly from a new audio recording, so uploading a new voice sample is not supported reliably. Live audio input would need a model trained on raw-audio datasets with a consistent feature extractor.
- Single dataset, single task (sustained vowel), no external validation.
- Not clinically validated.

## Project structure

```
.
├── parkinson_voice_model.py   # full pipeline
├── requirements.txt
├── README.md
├── pd_speech_features.csv     # download separately (not in repo)
└── voice_outputs/             # generated results
```

## Reference

Sakar, C. O., Serbes, G., Gunduz, A., Tunc, H. C., Nizam, H., Sakar, B. E., Tutuncu, M., Aydin, T., Isenkul, M. E., Apaydin, H. (2019). *A comparative analysis of speech signal processing algorithms for Parkinson's disease classification and the use of the tunable Q-factor wavelet transform.* Applied Soft Computing, 74, 255-263.

## Authors

Voice module by <your name>, part of a team project (sleep, handwriting and gait modules by teammates).