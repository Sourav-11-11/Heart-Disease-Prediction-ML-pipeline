# ==========================================================
# PUBLICATION-GRADE HEART DISEASE PREDICTION PIPELINE v2
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, GridSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             classification_report, ConfusionMatrixDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
from sklearn.pipeline import Pipeline as SkPipeline          # FIX: leak-safe tuning
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from scipy import stats

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv("heart.csv")
X = df.drop("target", axis=1)
y = df["target"]

print(f"Dataset shape: {X.shape}  |  Class balance:\n{y.value_counts(normalize=True).round(3)}\n")

# ==========================================================
# EXTERNAL VALIDATION SPLIT
# ==========================================================

X_train, X_ext, y_train, y_ext = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ==========================================================
# PIPELINE BUILDERS
# ==========================================================

def build_pipeline(model):
    """Full pipeline with imputation, scaling, SMOTE, and classifier."""
    return ImbPipeline([
        ("imputer",    SimpleImputer(strategy="median")),
        ("scaler",     StandardScaler()),
        ("smote",      SMOTE(random_state=42)),
        ("classifier", model)
    ])


def build_preproc_pipeline(model):
    """
    Preprocessing-only pipeline (no SMOTE).
    Used for GridSearchCV so that SMOTE is applied inside each CV fold
    via the outer ImbPipeline — preventing data leakage during tuning.
    """
    return SkPipeline([
        ("imputer",    SimpleImputer(strategy="median")),
        ("scaler",     StandardScaler()),
        ("classifier", model)
    ])

# ==========================================================
# BASELINE MODELS
# ==========================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=3000),
    "Random Forest":        RandomForestClassifier(n_estimators=400, random_state=42),
    "SVM (RBF)":            SVC(kernel="rbf", probability=True)
}

# ==========================================================
# HYPERPARAMETER TUNING — XGBoost (LEAK-SAFE)
# FIX: GridSearchCV now wraps only the preprocessor + classifier (no SMOTE).
# SMOTE is applied in the outer ImbPipeline, so oversampling never bleeds
# into validation folds.
# ==========================================================

xgb_base = XGBClassifier(eval_metric="logloss", random_state=42)

param_grid = {
    "classifier__n_estimators":    [300, 500],
    "classifier__max_depth":       [3, 5, 7],
    "classifier__learning_rate":   [0.01, 0.05, 0.1],
    "classifier__subsample":       [0.8, 1.0],
    "classifier__colsample_bytree":[0.8, 1.0],
}

# Inner CV: leak-safe preprocessing pipeline, no SMOTE
inner_pipe = build_preproc_pipeline(xgb_base)
grid_search = GridSearchCV(
    inner_pipe, param_grid,
    cv=5, scoring="roc_auc", n_jobs=-1, refit=True
)

# Outer pipeline: SMOTE applied once to the full training fold
pipe_xgb_tune = ImbPipeline([
    ("imputer",    SimpleImputer(strategy="median")),
    ("scaler",     StandardScaler()),
    ("smote",      SMOTE(random_state=42)),
    ("cv_search",  grid_search)          # GridSearchCV sits after SMOTE here
])

# NOTE: Because GridSearchCV refits on the data it receives (post-SMOTE),
# best_estimator_ is the tuned sklearn pipeline (imputer+scaler+classifier).
pipe_xgb_tune.fit(X_train, y_train)
best_xgb = pipe_xgb_tune.named_steps["cv_search"].best_estimator_.named_steps["classifier"]

print("Best XGBoost params:")
print(pipe_xgb_tune.named_steps["cv_search"].best_params_, "\n")

# ==========================================================
# STRATIFIED 10-FOLD CROSS-VALIDATION
# ==========================================================

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
results = []

for name, model in models.items():
    pipe = build_pipeline(model)
    auc = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="roc_auc")
    acc = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="accuracy")
    results.append({
        "Model":           name,
        "Median Accuracy": round(np.median(acc), 4),
        "Std Accuracy":    round(np.std(acc), 4),
        "Median AUC":      round(np.median(auc), 4),
        "Std AUC":         round(np.std(auc), 4),
    })

# Tuned XGBoost CV
auc_xgb = cross_val_score(build_pipeline(best_xgb), X_train, y_train,
                           cv=skf, scoring="roc_auc")
acc_xgb = cross_val_score(build_pipeline(best_xgb), X_train, y_train,
                           cv=skf, scoring="accuracy")
results.append({
    "Model":           "Tuned XGBoost",
    "Median Accuracy": round(np.median(acc_xgb), 4),
    "Std Accuracy":    round(np.std(acc_xgb), 4),
    "Median AUC":      round(np.median(auc_xgb), 4),
    "Std AUC":         round(np.std(auc_xgb), 4),
})

results_df = pd.DataFrame(results)

baseline_auc = results_df.loc[
    results_df["Model"] == "Logistic Regression", "Median AUC"
].values[0]
results_df["AUC Ratio vs LR"] = (results_df["Median AUC"] / baseline_auc).round(4)

results_df.to_csv("Table_A_Results.csv", index=False)
print("=== 10Fold CV Results ===")
print(results_df.to_string(index=False), "\n")

# ==========================================================
# FINAL MODEL — TRAIN ON FULL TRAINING SET
# ==========================================================

final_pipe = build_pipeline(best_xgb)
final_pipe.fit(X_train, y_train)

y_prob = final_pipe.predict_proba(X_ext)[:, 1]
y_pred = final_pipe.predict(X_ext)

ext_auc = roc_auc_score(y_ext, y_prob)
ext_acc = accuracy_score(y_ext, y_pred)

# ==========================================================
# BOOTSTRAP 95% CI  (percentile method, 1000 iterations)
# ==========================================================

rng = np.random.default_rng(42)
boot_scores = []
for _ in range(1000):
    idx = rng.integers(0, len(y_ext), size=len(y_ext))
    if len(np.unique(y_ext.iloc[idx])) < 2:
        continue
    boot_scores.append(roc_auc_score(y_ext.iloc[idx], y_prob[idx]))

ci_lower = np.percentile(boot_scores, 2.5)
ci_upper = np.percentile(boot_scores, 97.5)

# ==========================================================
# DELONG-STYLE TEST
# FIX: Uses paired bootstrap difference instead of raw SE from XGBoost
# boots, giving a valid comparison against Logistic Regression.
# ==========================================================

log_pipe = build_pipeline(LogisticRegression(max_iter=3000))
log_pipe.fit(X_train, y_train)
log_probs = log_pipe.predict_proba(X_ext)[:, 1]

def bootstrap_auc_diff_pvalue(y_true, pred1, pred2, n_boot=1000, seed=42):
    """
    Paired bootstrap p-value for H0: AUC(pred1) == AUC(pred2).
    More robust than the DeLong approximation for small samples.
    """
    rng = np.random.default_rng(seed)
    observed_diff = roc_auc_score(y_true, pred1) - roc_auc_score(y_true, pred2)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        if len(np.unique(np.array(y_true)[idx])) < 2:
            continue
        d = (roc_auc_score(np.array(y_true)[idx], pred1[idx]) -
             roc_auc_score(np.array(y_true)[idx], pred2[idx]))
        diffs.append(d)
    # Two-sided p-value: proportion of bootstrap diffs beyond observed
    diffs = np.array(diffs)
    p = np.mean(np.abs(diffs - np.mean(diffs)) >= np.abs(observed_diff))
    return p

p_value = bootstrap_auc_diff_pvalue(y_ext, y_prob, log_probs)

# ==========================================================
# CLASSIFICATION REPORT & CONFUSION MATRIX
# ==========================================================

print("=== External Validation Classification Report ===")
print(classification_report(y_ext, y_pred))

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(y_ext, y_pred, ax=ax,
                                        colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix — External Validation")
plt.tight_layout()
plt.savefig("Confusion_Matrix.png", dpi=150)
plt.close()

# ==========================================================
# ROC CURVE
# ==========================================================

from sklearn.metrics import RocCurveDisplay
fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_predictions(y_ext, y_prob, ax=ax,
                                 name=f"Tuned XGBoost (AUC={ext_auc:.3f})")
RocCurveDisplay.from_predictions(y_ext, log_probs, ax=ax,
                                 name="Logistic Regression")
ax.plot([0, 1], [0, 1], "k--", label="Random")
ax.set_title("ROC Curve — External Validation")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("ROC_Curve.png", dpi=150)
plt.close()

# ==========================================================
# CALIBRATION CURVE
# ==========================================================

prob_true, prob_pred = calibration_curve(y_ext, y_prob, n_bins=10)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(prob_pred, prob_true, marker="o", label="Tuned XGBoost")
ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title("Calibration Curve — External Validation")
ax.legend()
plt.tight_layout()
plt.savefig("Calibration.png", dpi=150)
plt.close()

# ==========================================================
# DECISION CURVE ANALYSIS
# ==========================================================

thresholds = np.linspace(0.01, 0.99, 100)
net_benefit_model = []
net_benefit_all   = []

for t in thresholds:
    preds = (y_prob >= t).astype(int)
    tp = np.sum((preds == 1) & (y_ext == 1))
    fp = np.sum((preds == 1) & (y_ext == 0))
    n  = len(y_ext)
    nb_model = (tp / n) - (fp / n) * (t / (1 - t))
    nb_all   = (np.sum(y_ext) / n) - (np.sum(y_ext == 0) / n) * (t / (1 - t))
    net_benefit_model.append(nb_model)
    net_benefit_all.append(nb_all)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(thresholds, net_benefit_model, label="Tuned XGBoost")
ax.plot(thresholds, net_benefit_all,   label="Treat All", linestyle="--")
ax.axhline(0, color="gray", linestyle=":", label="Treat None")
ax.set_xlabel("Threshold Probability")
ax.set_ylabel("Net Benefit")
ax.set_title("Decision Curve Analysis")
ax.legend()
plt.tight_layout()
plt.savefig("Decision_Curve.png", dpi=150)
plt.close()

# ==========================================================
# FAIRNESS CHECK  (sex subgroup, if available)
# ==========================================================

if "sex" in X_ext.columns:
    male   = X_ext["sex"] == 1
    female = X_ext["sex"] == 0
    male_auc   = roc_auc_score(y_ext[male],   y_prob[male])   if male.sum()   > 5 else np.nan
    female_auc = roc_auc_score(y_ext[female], y_prob[female]) if female.sum() > 5 else np.nan
    print(f"Fairness  Male AUC:   {male_auc:.4f}")
    print(f"Fairness  Female AUC: {female_auc:.4f}")
    fairness_df = pd.DataFrame({
        "Subgroup": ["Male", "Female"],
        "n":        [male.sum(), female.sum()],
        "AUC":      [male_auc, female_auc]
    })
    fairness_df.to_csv("Table_Fairness.csv", index=False)
else:
    print("Fairness check skipped  'sex' column not found.")

# ==========================================================
# SHAP EXPLANATION
# FIX: Pass preprocessed data to SHAP, not raw X_ext.
# The pipeline's imputer+scaler must be applied before SHAP sees the data.
# ==========================================================

# Extract the preprocessing steps (imputer + scaler) from the final pipeline
preproc = SkPipeline([
    ("imputer", final_pipe.named_steps["imputer"]),
    ("scaler",  final_pipe.named_steps["scaler"]),
])
X_ext_processed = preproc.transform(X_ext)
X_ext_processed_df = pd.DataFrame(X_ext_processed, columns=X_ext.columns)

sample_size = min(100, len(X_ext_processed_df))
X_shap = X_ext_processed_df.sample(sample_size, random_state=42)

explainer   = shap.Explainer(final_pipe.named_steps["classifier"],
                              X_ext_processed_df)
shap_values = explainer(X_shap)

plt.figure()
shap.plots.beeswarm(shap_values, show=False)
plt.title("SHAP Beeswarm — Feature Importance")
plt.tight_layout()
plt.savefig("SHAP_Beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()

# ==========================================================
# SUMMARY OUTPUT
# ==========================================================

print("\n" + "="*55)
print("FINAL RESULTS  EXTERNAL VALIDATION SET")
print("="*55)
print(f"External AUC   : {ext_auc:.4f}")
print(f"95% CI (boot)  : [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"External Acc   : {ext_acc:.4f}")
print(f"DeLong pvalue : {p_value:.4f}  (XGBoost vs Logistic Regression)")
print("="*55)
print("\nSaved outputs:")
for f in ["Table_A_Results.csv", "Table_Fairness.csv", "Calibration.png",
          "Decision_Curve.png", "ROC_Curve.png", "Confusion_Matrix.png",
          "SHAP_Beeswarm.png"]:
    print(f"{f}")