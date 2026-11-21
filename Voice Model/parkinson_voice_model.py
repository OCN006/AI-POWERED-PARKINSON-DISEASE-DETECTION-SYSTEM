# =============================================================================
# Parkinson's Disease - VOICE MODEL (UCI Speech Features dataset, id=470)
# Runs as a .py script OR paste cell-by-cell into Jupyter / Colab (# %% = cell)
#
# Install once:  pip install pandas numpy scikit-learn matplotlib joblib xgboost shap
# Put pd_speech_features.csv in the same folder as this file.
# =============================================================================

# %% 0. Setup ------------------------------------------------------------------
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             matthews_corrcoef, confusion_matrix, roc_curve,
                             ConfusionMatrixDisplay)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    print("xgboost not found -> using ExtraTrees instead (pip install xgboost to enable)")

DATA_PATH = "pd_speech_features.csv"
OUT_DIR = "voice_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42
np.random.seed(SEED)

# %% 1. Load data --------------------------------------------------------------
# The UCI file has a group-header row first, so skiprows=1 is required.
df = pd.read_csv(DATA_PATH, skiprows=1)
print("Shape:", df.shape)                       # expect (756, 755)
print("Subjects:", df["id"].nunique())          # expect 252
print("Missing values:", int(df.isna().sum().sum()))
print(df["class"].value_counts().rename({1: "PD (1)", 0: "Healthy (0)"}))

df = df.fillna(df.median(numeric_only=True))    # safety net (dataset has none)

groups = df["id"].values                        # subject id -> used for SPLITTING
y = df["class"].values.astype(int)
feature_cols = [c for c in df.columns if c not in ("id", "class")]
X = df[feature_cols].values.astype(float)
print("Feature matrix:", X.shape)

# %% 2. Helpers ----------------------------------------------------------------
def make_pipe(clf, k=100):
    """Scaling + selection live INSIDE the pipeline -> fit on train folds only (no leakage)."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("var", VarianceThreshold(0.0)),         # drop constant features
        ("select", SelectKBest(f_classif, k=k)),
        ("clf", clf),
    ])


def subject_level(p, y_, g):
    """Average the 3 recordings of each person -> one prediction per subject."""
    d = pd.DataFrame({"id": g, "y": y_, "p": p}).groupby("id").agg(y=("y", "first"), p=("p", "mean"))
    return d["y"].values.astype(int), d["p"].values


def metrics(y_true, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "BalancedAcc": balanced_accuracy_score(y_true, pred),
        "F1": f1_score(y_true, pred, zero_division=0),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Sensitivity": recall_score(y_true, pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "AUC": roc_auc_score(y_true, p),
        "MCC": matthews_corrcoef(y_true, pred),
    }


def oof_proba(model, X_, y_, g_, seed, n_splits=5):
    """Out-of-fold probabilities with SUBJECT-WISE folds (same person never in train+test)."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    p = np.zeros(len(y_))
    for tr, te in cv.split(X_, y_, g_):
        m = clone(model).fit(X_[tr], y_[tr])
        p[te] = m.predict_proba(X_[te])[:, 1]
    return p


def selected_names(pipe, cols):
    cols = np.array(cols)
    cols = cols[pipe.named_steps["var"].get_support()]
    return cols[pipe.named_steps["select"].get_support()]


def build_models(svm_params=None, k_svm=100):
    svm_params = svm_params or {"C": 1, "gamma": "scale"}
    svm = make_pipe(SVC(kernel="rbf", probability=True, class_weight="balanced",
                        random_state=SEED, **svm_params), k=k_svm)
    lr = make_pipe(LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000), k=100)
    rf = make_pipe(RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                          n_jobs=-1, random_state=SEED), k=100)
    if HAS_XGB:
        pos_w = (y == 0).sum() / (y == 1).sum()
        gb = make_pipe(XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.5, scale_pos_weight=pos_w,
                                     eval_metric="logloss", n_jobs=-1, random_state=SEED), k=100)
    else:
        gb = make_pipe(ExtraTreesClassifier(n_estimators=400, class_weight="balanced",
                                            n_jobs=-1, random_state=SEED), k=100)
    ens = VotingClassifier([("svm", clone(svm)), ("lr", clone(lr)), ("gb", clone(gb))], voting="soft")
    return {"SVM": svm, "LogReg": lr, "RandomForest": rf, "XGBoost" if HAS_XGB else "ExtraTrees": gb,
            "Ensemble(SVM+LR+GB)": ens}


# %% 3. Compare models with repeated subject-wise 5-fold CV --------------------
# Reported at SUBJECT level (3 recordings averaged). 3 repeats with different splits.
SEEDS = [0, 1, 2]
models = build_models()
rows = []
for name, mdl in models.items():
    per_seed = []
    for s in SEEDS:
        p = oof_proba(mdl, X, y, groups, seed=s)
        ys, ps = subject_level(p, y, groups)
        per_seed.append(metrics(ys, ps))
    r = pd.DataFrame(per_seed)
    rows.append({"Model": name, **{c: f"{r[c].mean():.3f} +/- {r[c].std():.3f}" for c in r.columns}})
    print(f"done: {name:22s} subject-level acc = {r['Accuracy'].mean():.3f}  bal.acc = {r['BalancedAcc'].mean():.3f}")

cv_table = pd.DataFrame(rows).set_index("Model")
print("\n=== Subject-wise CV results (subject level) ===")
print(cv_table.to_string())
cv_table.to_csv(f"{OUT_DIR}/cv_results.csv")

# %% 4. Hold-out subjects + tune SVM on the dev part only ----------------------
# ~20% of SUBJECTS are locked away; tuning never sees them.
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
dev_idx, test_idx = next(splitter.split(X, y, groups))
X_dev, y_dev, g_dev = X[dev_idx], y[dev_idx], groups[dev_idx]
X_te, y_te, g_te = X[test_idx], y[test_idx], groups[test_idx]
print(f"\nDev subjects: {len(set(g_dev))} | Hold-out subjects: {len(set(g_te))}")

tune_pipe = make_pipe(SVC(kernel="rbf", class_weight="balanced", random_state=SEED))
param_grid = {
    "select__k": [50, 100, 200],
    "clf__C": [0.5, 1, 5, 10],
    "clf__gamma": ["scale", 0.001, 0.003],
}
inner_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
gs = GridSearchCV(tune_pipe, param_grid, cv=inner_cv, scoring="balanced_accuracy", n_jobs=-1)
gs.fit(X_dev, y_dev, groups=g_dev)
print("Best SVM params:", gs.best_params_, "| inner CV balanced acc:", round(gs.best_score_, 3))

best_k = gs.best_params_["select__k"]
best_svm_params = {"C": gs.best_params_["clf__C"], "gamma": gs.best_params_["clf__gamma"]}
tuned = build_models(best_svm_params, best_k)

# %% 5. Evaluate on hold-out subjects (one-time, honest test) ------------------
holdout_rows, holdout_probs = [], {}
for name in ["SVM", "Ensemble(SVM+LR+GB)"]:
    m = clone(tuned[name]).fit(X_dev, y_dev)
    p = m.predict_proba(X_te)[:, 1]
    ys, ps = subject_level(p, y_te, g_te)
    holdout_probs[name] = (ys, ps)
    holdout_rows.append({"Model": name, **{k: round(v, 3) for k, v in metrics(ys, ps).items()}})
holdout_table = pd.DataFrame(holdout_rows).set_index("Model")
print("\n=== Hold-out subjects (tuned models, subject level) ===")
print(holdout_table.to_string())
holdout_table.to_csv(f"{OUT_DIR}/holdout_results.csv")

# %% 6. Plots: ROC + confusion matrix -------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for name, (ys, ps) in holdout_probs.items():
    fpr, tpr, _ = roc_curve(ys, ps)
    ax[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(ys, ps):.2f})")
ax[0].plot([0, 1], [0, 1], "k--", lw=1)
ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC - hold-out subjects")
ax[0].legend(loc="lower right")
ys, ps = holdout_probs["Ensemble(SVM+LR+GB)"]
ConfusionMatrixDisplay.from_predictions(ys, (ps >= 0.5).astype(int), display_labels=["Healthy", "PD"],
                                        ax=ax[1], colorbar=False)
ax[1].set_title("Ensemble - confusion matrix (subjects)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_confusion.png", dpi=200)
plt.show()

# %% 7. Feature importance (tree model on all data) + optional SHAP -------------
imp_pipe = clone(build_models(best_svm_params, best_k)["RandomForest"]).fit(X, y)
names = selected_names(imp_pipe, feature_cols)
imp = pd.Series(imp_pipe.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
imp.head(20)[::-1].plot.barh(figsize=(8, 6), title="Top 20 features (Random Forest importance)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=200)
plt.show()
imp.head(30).to_csv(f"{OUT_DIR}/top_features.csv")

try:  # SHAP is optional - script keeps going if it fails
    import shap
    Xt = imp_pipe[:-1].transform(X)
    sv = shap.TreeExplainer(imp_pipe.named_steps["clf"]).shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:
        sv = sv[:, :, 1]
    shap.summary_plot(sv, Xt, feature_names=list(names), max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/shap_summary.png", dpi=200)
    plt.show()
except Exception as e:
    print("SHAP skipped:", e)

# %% 8. Train FINAL model on all data, save, and integration helper -------------
final_model = clone(tuned["Ensemble(SVM+LR+GB)"]).fit(X, y)
joblib.dump({"model": final_model, "feature_cols": feature_cols}, f"{OUT_DIR}/voice_model.pkl")
print("Saved ->", f"{OUT_DIR}/voice_model.pkl")


def predict_voice(rows_df, model_path=f"{OUT_DIR}/voice_model.pkl"):
    """
    rows_df: DataFrame with the 754 feature columns (all recordings of ONE person).
    Returns probability of Parkinson's (0-1) for your team's fusion step.
    """
    bundle = joblib.load(model_path)
    p = bundle["model"].predict_proba(rows_df[bundle["feature_cols"]].values.astype(float))[:, 1].mean()
    return float(p)


# demo: pick one subject from the dataset
demo_id = df["id"].iloc[0]
prob = predict_voice(df[df["id"] == demo_id])
print(f"Demo subject {demo_id}: P(Parkinson's) = {prob:.3f} | true label = {int(df[df['id']==demo_id]['class'].iloc[0])}")
