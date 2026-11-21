# Dual-Sensor Sleep Staging CRNN

PyTorch pipeline for six-class sleep-stage classification from instantaneous heart-rate and accelerometry signals in the [BIDSleep dataset](https://physionet.org/content/bidsleep/1.0.0/).

The model predicts one clinical 30-second epoch at a time using two synchronized sensor streams:

| Class | Stage |
| ---: | --- |
| 0 | Wake |
| 1 | N1 |
| 2 | N2 |
| 3 | N3 |
| 4 | REM |
| 5 | Unscored / sensor error |

## Model Architecture

`DualSensorFusionModel` uses feature-level fusion:

1. Heart-rate and motion are processed by separate, identical sensor encoders.
2. Each encoder contains two 1D convolutional blocks followed by a bidirectional LSTM and mean temporal pooling.
3. The two 64-dimensional sensor representations are concatenated into a 128-dimensional feature vector.
4. A fully connected classifier with dropout produces six class logits.

### Input preprocessing

- Labels and signals are read from each `labels.mat`, `hr.csv`, and `motion.csv` recording.
- Each recording is normalized over the complete night with z-score normalization before epoch slicing.
- Signals are divided into 30-second epochs, truncated or zero-padded to 400 samples, and represented as `[batch, channel, length]` tensors.
- Only labels in the range `0` through `5` are included.

## Subject-Level Train/Test Split

The pipeline prevents subject leakage. Each epoch stores the subject ID from its top-level `BidslabXX` directory, and `GroupShuffleSplit` assigns complete subjects to exactly one partition:

- Training: `80%` of subjects
- Test: `20%` of subjects
- Random seed: `42`

All nights and epochs belonging to one subject remain in the same partition. The same deterministic split is recreated by `evaluate.py`, so the saved model is evaluated on the identical held-out subjects used during training.

This is a subject-level split rather than an epoch-level random split. Therefore, the test set size is determined by the number of epochs recorded for the held-out subjects and is reported when the scripts run.

With the dataset currently present in this repository, the seeded split contains 47 subjects and 218,439 valid epochs: 37 training subjects / 169,750 epochs and 10 test subjects / 48,689 epochs. These counts should be treated as a reproducibility record for this dataset version, not as a hard-coded assumption.

## Training

Training uses AdamW with a learning rate of `1e-3`, weight decay of `1e-4`, 15 epochs, and cosine-annealed learning rate scheduling. To reduce the effect of sleep-stage imbalance, balanced class weights are computed from the training subjects and clipped to the range `1.0` to `5.0` before being passed to cross-entropy loss.

When CUDA is available, `main.py`, `evaluate.py`, `evaluate_external.py`, and `predict.py` automatically place the model, batches, and inference tensors on the first CUDA device. Dataset directory scanning and CSV/MAT parsing remain CPU operations; the neural-network computation runs on CUDA.

```bash
python main.py
```

The trained weights are saved as `bidslab_sleep_model.pth`.

## Evaluation Metrics

Run detailed evaluation on the deterministic subject-level test partition. The checkpoint and dataset paths have sensible defaults, but both can be overridden from the command line:

```bash
python evaluate.py --checkpoint bidslab_sleep_model.pth \
	--data-path Sleepdata/a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0
```

On Windows PowerShell, use one line or PowerShell's backtick for continuation:

```powershell
\.venv\Scripts\python.exe evaluate.py --checkpoint bidslab_sleep_model.pth --data-path "Sleepdata\a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0" --output evaluation_report.json
```

Available options are `--checkpoint`, `--data-path`, `--batch-size`, and `--output`. The optional `--output` argument saves the completed metrics as JSON. Missing files, empty datasets, invalid batch sizes, and checkpoint/model shape mismatches produce a clear error message instead of a traceback.

The evaluator reports:

- **Accuracy:** fraction of correctly classified epochs.
- **Precision:** among epochs predicted as a class, the fraction that belong to that class.
- **Recall:** among epochs belonging to a class, the fraction correctly identified.
- **F1-score:** harmonic mean of precision and recall.
- **Macro ROC-AUC:** one-vs-rest ROC-AUC averaged equally across the six classes, computed from the model's softmax probabilities. It is shown as unavailable if the held-out subjects do not contain every class required for the calculation.
- **Confusion matrix:** six-by-six matrix with true classes as rows and predicted classes as columns.

The classification report includes per-class scores plus macro and weighted averages. Metrics should be reported from a completed run of `evaluate.py`; no fixed accuracy or F1 value is claimed in this README because those values depend on the locally available dataset and checkpoint.

## External Evaluation

To evaluate a completely separate labeled dataset, update `NEW_DATASET_PATH` in `evaluate_external.py` and run:

```bash
python evaluate_external.py
```

This path is treated as test-only data and is not included in training. The same precision/recall/F1 classification report and confusion matrix are produced for the external subjects.

## Repository Files

- `main.py`: dataset loader, model definition, subject-level training split, training loop, and training-set evaluation.
- `evaluate.py`: CLI evaluator that reloads the checkpoint and evaluates the held-out subjects with detailed metrics and optional JSON output.
- `evaluate_external.py`: evaluates labeled data from an external folder.
- `predict.py`: generates a hypnogram for raw heart-rate and motion CSV files without labels.
