# ==========================================================
# HEART DISEASE PREDICTION — STREAMLIT APP
# Run: streamlit run src/app.py
# Requires: data/heart.csv in the parent directory
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import shap
import warnings
import os
warnings.filterwarnings("ignore")

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, GridSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             classification_report, ConfusionMatrixDisplay,
                             RocCurveDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from scipy import stats
import io

# ----------------------------------------------------------
# SETUP PATHS
# ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------
# CUSTOM STYLES  (clinical / medical-grade dark aesthetic)
# ----------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}

/* Cards */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 20px 24px;
    text-align: center;
}
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #8b949e;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #58a6ff;
}
.metric-sub {
    font-size: 0.78rem;
    color: #8b949e;
    margin-top: 4px;
}

/* Section headers */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #58a6ff;
    border-bottom: 1px solid #21262d;
    padding-bottom: 8px;
    margin-bottom: 20px;
    margin-top: 32px;
}

/* Alert boxes */
.alert-green  { background:#0d2b1e; border:1px solid #2ea043; border-radius:6px; padding:12px 16px; color:#3fb950; }
.alert-yellow { background:#2b2200; border:1px solid #9e6a03; border-radius:6px; padding:12px 16px; color:#e3b341; }
.alert-red    { background:#2b0d0d; border:1px solid #f85149; border-radius:6px; padding:12px 16px; color:#f85149; }

/* Prediction badge */
.pred-positive {
    display:inline-block; background:#0d2b1e; border:2px solid #2ea043;
    border-radius:24px; padding:10px 28px;
    font-family:'IBM Plex Mono',monospace; font-size:1.1rem;
    color:#3fb950; font-weight:600;
}
.pred-negative {
    display:inline-block; background:#2b0d0d; border:2px solid #f85149;
    border-radius:24px; padding:10px 28px;
    font-family:'IBM Plex Mono',monospace; font-size:1.1rem;
    color:#f85149; font-weight:600;
}
.stButton>button {
    background: #1f6feb; color: #fff; border: none;
    border-radius: 6px; padding: 10px 24px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem; letter-spacing: 0.05em;
    transition: background 0.2s;
}
.stButton>button:hover { background: #388bfd; }

/* Tabs */
div[data-baseweb="tab-list"] { gap: 4px; }
div[data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# MATPLOTLIB THEME  (dark, clinical)
# ----------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#8b949e",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#e6edf3",
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "font.family":       "monospace",
    "figure.dpi":        130,
})

# ----------------------------------------------------------
# HELPER: convert matplotlib figure → Streamlit
# ----------------------------------------------------------
def fig_to_st(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf

# ----------------------------------------------------------
# PIPELINE BUILDERS
# ----------------------------------------------------------
def build_pipeline(model):
    return ImbPipeline([
        ("imputer",    SimpleImputer(strategy="median")),
        ("scaler",     StandardScaler()),
        ("smote",      SMOTE(random_state=42)),
        ("classifier", model)
    ])

def build_preproc_pipeline(model):
    return SkPipeline([
        ("imputer",    SimpleImputer(strategy="median")),
        ("scaler",     StandardScaler()),
        ("classifier", model)
    ])

# ----------------------------------------------------------
# CACHED TRAINING  (only re-runs when data changes)
# ----------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_all(df_hash: int, df: pd.DataFrame):
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_ext, y_train, y_ext = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── Baseline models ──────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000),
        "Random Forest":        RandomForestClassifier(n_estimators=400, random_state=42),
        "SVM (RBF)":            SVC(kernel="rbf", probability=True),
    }

    # ── XGBoost tuning (leak-safe) ────────────────────────
    xgb_base = XGBClassifier(eval_metric="logloss", random_state=42)
    param_grid = {
        "classifier__n_estimators":     [300, 500],
        "classifier__max_depth":        [3, 5],
        "classifier__learning_rate":    [0.05, 0.1],
        "classifier__subsample":        [0.8, 1.0],
        "classifier__colsample_bytree": [0.8, 1.0],
    }
    inner_pipe  = build_preproc_pipeline(xgb_base)
    grid_search = GridSearchCV(inner_pipe, param_grid, cv=5,
                               scoring="roc_auc", n_jobs=-1, refit=True)
    pipe_xgb_tune = ImbPipeline([
        ("imputer",   SimpleImputer(strategy="median")),
        ("scaler",    StandardScaler()),
        ("smote",     SMOTE(random_state=42)),
        ("cv_search", grid_search),
    ])
    pipe_xgb_tune.fit(X_train, y_train)
    best_xgb = (pipe_xgb_tune.named_steps["cv_search"]
                              .best_estimator_
                              .named_steps["classifier"])
    best_params = pipe_xgb_tune.named_steps["cv_search"].best_params_

    # ── 10-Fold CV ────────────────────────────────────────
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = []
    for name, model in models.items():
        pipe = build_pipeline(model)
        auc  = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="roc_auc")
        acc  = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="accuracy")
        results.append({
            "Model":           name,
            "Median Accuracy": round(np.median(acc), 4),
            "Std Accuracy":    round(np.std(acc),    4),
            "Median AUC":      round(np.median(auc), 4),
            "Std AUC":         round(np.std(auc),    4),
        })
    auc_xgb = cross_val_score(build_pipeline(best_xgb), X_train, y_train,
                               cv=skf, scoring="roc_auc")
    acc_xgb = cross_val_score(build_pipeline(best_xgb), X_train, y_train,
                               cv=skf, scoring="accuracy")
    results.append({
        "Model":           "Tuned XGBoost",
        "Median Accuracy": round(np.median(acc_xgb), 4),
        "Std Accuracy":    round(np.std(acc_xgb),    4),
        "Median AUC":      round(np.median(auc_xgb), 4),
        "Std AUC":         round(np.std(auc_xgb),    4),
    })
    results_df = pd.DataFrame(results)
    baseline_auc = results_df.loc[
        results_df["Model"] == "Logistic Regression", "Median AUC"
    ].values[0]
    results_df["AUC Ratio vs LR"] = (results_df["Median AUC"] / baseline_auc).round(4)

    # ── Final model ───────────────────────────────────────
    final_pipe = build_pipeline(best_xgb)
    final_pipe.fit(X_train, y_train)

    y_prob = final_pipe.predict_proba(X_ext)[:, 1]
    y_pred = final_pipe.predict(X_ext)
    ext_auc = roc_auc_score(y_ext, y_prob)
    ext_acc = accuracy_score(y_ext, y_pred)

    # ── Bootstrap CI ──────────────────────────────────────
    rng = np.random.default_rng(42)
    boot_scores = []
    for _ in range(1000):
        idx = rng.integers(0, len(y_ext), size=len(y_ext))
        if len(np.unique(y_ext.iloc[idx])) < 2:
            continue
        boot_scores.append(roc_auc_score(y_ext.iloc[idx], y_prob[idx]))
    ci_lower = np.percentile(boot_scores, 2.5)
    ci_upper = np.percentile(boot_scores, 97.5)

    # ── Paired bootstrap p-value (XGBoost vs LR) ─────────
    log_pipe = build_pipeline(LogisticRegression(max_iter=3000))
    log_pipe.fit(X_train, y_train)
    log_probs = log_pipe.predict_proba(X_ext)[:, 1]

    def bootstrap_auc_diff_pvalue(y_true, p1, p2, n_boot=1000, seed=42):
        rng2 = np.random.default_rng(seed)
        obs  = roc_auc_score(y_true, p1) - roc_auc_score(y_true, p2)
        diffs = []
        for _ in range(n_boot):
            idx = rng2.integers(0, len(y_true), size=len(y_true))
            if len(np.unique(np.array(y_true)[idx])) < 2:
                continue
            diffs.append(roc_auc_score(np.array(y_true)[idx], p1[idx]) -
                         roc_auc_score(np.array(y_true)[idx], p2[idx]))
        diffs = np.array(diffs)
        return float(np.mean(np.abs(diffs - np.mean(diffs)) >= np.abs(obs)))

    p_value = bootstrap_auc_diff_pvalue(y_ext, y_prob, log_probs)

    # ── SHAP (preprocessed) ───────────────────────────────
    preproc = SkPipeline([
        ("imputer", final_pipe.named_steps["imputer"]),
        ("scaler",  final_pipe.named_steps["scaler"]),
    ])
    X_ext_proc = pd.DataFrame(preproc.transform(X_ext), columns=X.columns)
    X_shap     = X_ext_proc.sample(min(100, len(X_ext_proc)), random_state=42)
    explainer   = shap.Explainer(final_pipe.named_steps["classifier"], X_ext_proc)
    shap_values = explainer(X_shap)

    # ── Fairness ──────────────────────────────────────────
    fairness = {}
    if "sex" in X_ext.columns:
        for grp, label in [(1, "Male"), (0, "Female")]:
            mask = X_ext["sex"] == grp
            if mask.sum() > 5:
                fairness[label] = round(roc_auc_score(y_ext[mask], y_prob[mask]), 4)

    return dict(
        X=X, y=y, X_train=X_train, X_ext=X_ext,
        y_train=y_train, y_ext=y_ext,
        final_pipe=final_pipe, log_pipe=log_pipe,
        y_prob=y_prob, log_probs=log_probs, y_pred=y_pred,
        ext_auc=ext_auc, ext_acc=ext_acc,
        ci_lower=ci_lower, ci_upper=ci_upper, p_value=p_value,
        results_df=results_df, best_params=best_params,
        shap_values=shap_values, X_shap=X_shap,
        fairness=fairness,
    )

# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:
    st.markdown("### 🫀 Heart Disease\nPrediction Pipeline")
    st.markdown("---")

    uploaded = st.file_uploader("Upload heart.csv", type=["csv"])
    st.markdown("---")

    st.markdown("**About**")
    st.caption(
        "Publication-grade ML pipeline with SMOTE, "
        "leak-safe XGBoost tuning, bootstrap CI, "
        "SHAP explanations, and fairness checks."
    )
    st.markdown("---")
    st.caption("v2.0 · Powered by XGBoost + Streamlit")

# ==========================================================
# LOAD DATA
# ==========================================================
if uploaded:
    df = pd.read_csv(uploaded)
else:
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "heart.csv"))
        st.sidebar.success("Loaded heart.csv from data folder")
    except FileNotFoundError:
        st.error("**heart.csv not found in data folder.** Please upload the file using the sidebar.")
        st.stop()

if "target" not in df.columns:
    st.error("Dataset must contain a 'target' column.")
    st.stop()

# ==========================================================
# TRAIN
# ==========================================================
with st.spinner("⚙️  Training models — this may take ~60 seconds on first run…"):
    r = train_all(hash(df.to_json()), df)

# ==========================================================
# HEADER
# ==========================================================
st.markdown(
    "<h1 style='font-family:IBM Plex Mono,monospace;font-size:1.6rem;"
    "color:#e6edf3;margin-bottom:4px;'>🫀 Heart Disease Prediction</h1>"
    "<p style='color:#8b949e;font-size:0.88rem;margin-top:0;'>"
    "Publication-Grade ML Pipeline · XGBoost · SHAP · Bootstrap CI</p>",
    unsafe_allow_html=True
)

# ==========================================================
# TOP METRIC CARDS
# ==========================================================
c1, c2, c3, c4 = st.columns(4)
for col, label, val, sub in [
    (c1, "External AUC",   f"{r['ext_auc']:.4f}",
         f"95% CI [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"),
    (c2, "External Accuracy", f"{r['ext_acc']:.4f}",
         f"On {len(r['y_ext'])} held-out samples"),
    (c3, "DeLong p-value", f"{r['p_value']:.4f}",
         "XGBoost vs Logistic Regression"),
    (c4, "Train / Test Split", f"80 / 20",
         f"{len(r['X_train'])} train · {len(r['X_ext'])} test"),
]:
    col.markdown(
        f"<div class='metric-card'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{val}</div>"
        f"<div class='metric-sub'>{sub}</div>"
        f"</div>", unsafe_allow_html=True
    )

# ==========================================================
# TABS
# ==========================================================
tabs = st.tabs([
    "📊 CV Results",
    "📈 ROC & Calibration",
    "🎯 Decision Curve",
    "🔍 SHAP",
    "⚖️ Fairness",
    "🩺 Predict",
    "🗂️ Data",
])

# ── Tab 0: CV Results ────────────────────────────────────
with tabs[0]:
    st.markdown("<div class='section-header'>10-Fold Cross-Validation Results</div>",
                unsafe_allow_html=True)

    df_res = r["results_df"].copy()
    st.dataframe(
        df_res.style
            .highlight_max(subset=["Median AUC", "Median Accuracy"],
                           color="#0d2b1e")
            .format({"Median AUC": "{:.4f}", "Median Accuracy": "{:.4f}",
                     "Std AUC":    "{:.4f}", "Std Accuracy":    "{:.4f}",
                     "AUC Ratio vs LR": "{:.4f}"}),
        use_container_width=True, hide_index=True
    )

    st.markdown("<div class='section-header'>AUC Comparison (Bar)</div>",
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    colors = ["#58a6ff" if m != "Tuned XGBoost" else "#3fb950"
              for m in df_res["Model"]]
    bars = ax.barh(df_res["Model"], df_res["Median AUC"],
                   color=colors, height=0.5, zorder=3)
    ax.errorbar(df_res["Median AUC"], df_res["Model"],
                xerr=df_res["Std AUC"],
                fmt="none", color="#e6edf3", capsize=4, linewidth=1.2, zorder=4)
    ax.set_xlabel("Median AUC (10-Fold CV)")
    ax.set_xlim(0.7, 1.0)
    ax.grid(axis="x", zorder=0)
    for bar, v in zip(bars, df_res["Median AUC"]):
        ax.text(v + 0.003, bar.get_y() + bar.get_height()/2,
                f"{v:.4f}", va="center", fontsize=8, color="#e6edf3")
    fig.tight_layout()
    st.image(fig_to_st(fig))

    st.markdown("<div class='section-header'>Best XGBoost Parameters</div>",
                unsafe_allow_html=True)
    params_df = pd.DataFrame(
        [(k.replace("classifier__", ""), v)
         for k, v in r["best_params"].items()],
        columns=["Parameter", "Value"]
    )
    st.dataframe(params_df, use_container_width=True, hide_index=True)

# ── Tab 1: ROC & Calibration ─────────────────────────────
with tabs[1]:
    col_roc, col_cal = st.columns(2)

    with col_roc:
        st.markdown("<div class='section-header'>ROC Curve</div>",
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        RocCurveDisplay.from_predictions(
            r["y_ext"], r["y_prob"], ax=ax,
            name=f"Tuned XGBoost (AUC={r['ext_auc']:.3f})",
            color="#3fb950"
        )
        RocCurveDisplay.from_predictions(
            r["y_ext"], r["log_probs"], ax=ax,
            name="Logistic Regression", color="#58a6ff"
        )
        ax.plot([0,1],[0,1],"--", color="#8b949e", label="Random")
        ax.fill_between([0,1],[0,1], alpha=0.04, color="#8b949e")
        ax.legend(fontsize=8)
        ax.set_title("ROC — External Validation")
        fig.tight_layout()
        st.image(fig_to_st(fig))

    with col_cal:
        st.markdown("<div class='section-header'>Calibration Curve</div>",
                    unsafe_allow_html=True)
        prob_true, prob_pred = calibration_curve(
            r["y_ext"], r["y_prob"], n_bins=10
        )
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(prob_pred, prob_true, "o-", color="#3fb950",
                linewidth=2, markersize=6, label="Tuned XGBoost")
        ax.plot([0,1],[0,1],"--", color="#8b949e", label="Perfect")
        ax.fill_between(prob_pred, prob_pred, prob_true,
                        alpha=0.15, color="#f85149", label="Calibration gap")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Curve")
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.image(fig_to_st(fig))

    st.markdown("<div class='section-header'>Confusion Matrix</div>",
                unsafe_allow_html=True)
    col_cm, col_cr = st.columns([1, 1.6])
    with col_cm:
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ConfusionMatrixDisplay.from_predictions(
            r["y_ext"], r["y_pred"], ax=ax,
            colorbar=False, cmap="YlOrRd",
            text_kw={"color": "#e6edf3", "fontsize": 13}
        )
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        st.image(fig_to_st(fig))
    with col_cr:
        st.markdown("**Classification Report**")
        report = classification_report(r["y_ext"], r["y_pred"], output_dict=True)
        report_df = pd.DataFrame(report).T.drop("support", axis=1).round(4)
        st.dataframe(report_df, use_container_width=True)

# ── Tab 2: Decision Curve ────────────────────────────────
with tabs[2]:
    st.markdown("<div class='section-header'>Decision Curve Analysis</div>",
                unsafe_allow_html=True)

    thresholds       = np.linspace(0.01, 0.99, 100)
    net_benefit_xgb  = []
    net_benefit_all  = []
    y_ext_arr        = np.array(r["y_ext"])
    y_prob_arr       = r["y_prob"]

    for t in thresholds:
        preds = (y_prob_arr >= t).astype(int)
        tp = np.sum((preds==1) & (y_ext_arr==1))
        fp = np.sum((preds==1) & (y_ext_arr==0))
        n  = len(y_ext_arr)
        net_benefit_xgb.append((tp/n) - (fp/n)*(t/(1-t)))
        net_benefit_all.append(
            (np.sum(y_ext_arr)/n) - (np.sum(y_ext_arr==0)/n)*(t/(1-t))
        )

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(thresholds, net_benefit_xgb,  color="#3fb950", lw=2,   label="Tuned XGBoost")
    ax.plot(thresholds, net_benefit_all,  color="#58a6ff", lw=1.5, linestyle="--", label="Treat All")
    ax.axhline(0, color="#8b949e", linestyle=":", lw=1.2, label="Treat None")
    ax.fill_between(thresholds,
                    np.array(net_benefit_xgb),
                    np.array(net_benefit_all),
                    where=np.array(net_benefit_xgb) > np.array(net_benefit_all),
                    alpha=0.12, color="#3fb950", label="Net gain region")
    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_title("Decision Curve Analysis — External Validation")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(-0.1, 0.7)
    fig.tight_layout()
    st.image(fig_to_st(fig))
    st.caption(
        "**Reading this chart**: Net Benefit = True Positive Rate − "
        "False Positive Rate × (threshold / (1 − threshold)). "
        "The model adds clinical value wherever its curve sits above both "
        "Treat All and Treat None."
    )

# ── Tab 3: SHAP ──────────────────────────────────────────
with tabs[3]:
    st.markdown("<div class='section-header'>SHAP Feature Importance</div>",
                unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.beeswarm(r["shap_values"], show=False, max_display=13,
                        color_bar=True)
    plt.title("SHAP Beeswarm — Top Feature Impacts", pad=12)
    plt.tight_layout()
    st.image(fig_to_st(plt.gcf()))
    plt.close("all")

    st.markdown("<div class='section-header'>Mean |SHAP| (Feature Ranking)</div>",
                unsafe_allow_html=True)
    mean_shap = np.abs(r["shap_values"].values).mean(axis=0)
    shap_df = (
        pd.DataFrame({"Feature": r["X_shap"].columns, "Mean |SHAP|": mean_shap})
        .sort_values("Mean |SHAP|", ascending=False)
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(shap_df["Feature"][::-1], shap_df["Mean |SHAP|"][::-1],
            color="#58a6ff", height=0.6)
    ax.set_xlabel("Mean |SHAP| Value")
    ax.set_title("Global Feature Importance")
    fig.tight_layout()
    st.image(fig_to_st(fig))

# ── Tab 4: Fairness ──────────────────────────────────────
with tabs[4]:
    st.markdown("<div class='section-header'>Subgroup Fairness Analysis</div>",
                unsafe_allow_html=True)

    if r["fairness"]:
        for grp, auc in r["fairness"].items():
            diff = abs(auc - r["ext_auc"])
            color_cls = "alert-green" if diff < 0.03 else \
                        "alert-yellow" if diff < 0.07 else "alert-red"
            st.markdown(
                f"<div class='{color_cls}'>"
                f"<strong>{grp}</strong> — AUC: <strong>{auc:.4f}</strong> "
                f"(Δ from overall: {diff:+.4f})"
                f"</div>", unsafe_allow_html=True
            )
            st.markdown("")

        if len(r["fairness"]) == 2:
            vals = list(r["fairness"].values())
            labels = list(r["fairness"].keys()) + ["Overall"]
            auc_vals = vals + [r["ext_auc"]]
            fig, ax = plt.subplots(figsize=(6, 3))
            colors = ["#58a6ff", "#f0883e", "#3fb950"]
            bars = ax.bar(labels, auc_vals, color=colors, width=0.45, zorder=3)
            ax.set_ylim(0.5, 1.0)
            ax.axhline(r["ext_auc"], color="#8b949e", linestyle="--",
                       lw=1, label="Overall AUC")
            ax.set_ylabel("AUC")
            ax.set_title("AUC by Sex Subgroup")
            for b, v in zip(bars, auc_vals):
                ax.text(b.get_x() + b.get_width()/2, v + 0.008,
                        f"{v:.4f}", ha="center", fontsize=9, color="#e6edf3")
            fig.tight_layout()
            st.image(fig_to_st(fig))
    else:
        st.markdown(
            "<div class='alert-yellow'>No 'sex' column detected — "
            "fairness analysis not available for this dataset.</div>",
            unsafe_allow_html=True
        )

# ── Tab 5: Predict ───────────────────────────────────────
with tabs[5]:
    st.markdown("<div class='section-header'>Single-Patient Prediction</div>",
                unsafe_allow_html=True)
    st.caption("Enter clinical values below. The model will return a risk probability.")

    feature_cols = list(r["X"].columns)
    col_defaults = r["X"].median().to_dict()

    # ── Detect binary columns (exactly 2 unique values) ──
    # Maps: feature → sorted list of its unique integer values
    binary_cols = {
        feat: sorted(r["X"][feat].dropna().unique().astype(int).tolist())
        for feat in feature_cols
        if r["X"][feat].dropna().nunique() == 2
    }

    # Human-readable labels for known heart-disease binary features
    BINARY_LABELS = {
        "sex":     {0: "0 — Female", 1: "1 — Male"},
        "fbs":     {0: "0 — False (≤120 mg/dl)", 1: "1 — True (>120 mg/dl)"},
        "exang":   {0: "0 — No", 1: "1 — Yes"},
        "target":  {0: "0 — No Disease", 1: "1 — Disease"},
    }

    input_cols = st.columns(3)
    user_input = {}
    for i, feat in enumerate(feature_cols):
        with input_cols[i % 3]:
            if feat in binary_cols:
                options = binary_cols[feat]
                label_map = BINARY_LABELS.get(feat, {v: str(v) for v in options})
                default_val = int(round(col_defaults.get(feat, options[0])))
                default_val = default_val if default_val in options else options[0]
                chosen_label = st.selectbox(
                    feat,
                    options=[label_map[v] for v in options],
                    index=options.index(default_val),
                    key=f"input_{feat}"
                )
                # Reverse map: label → numeric value
                rev_map = {v: k for k, v in label_map.items()}
                user_input[feat] = float(rev_map[chosen_label])
            else:
                user_input[feat] = st.number_input(
                    feat,
                    value=float(round(col_defaults.get(feat, 0.0), 2)),
                    format="%.2f",
                    key=f"input_{feat}"
                )

    if st.button("Run Prediction →"):
        input_df = pd.DataFrame([user_input])
        prob     = r["final_pipe"].predict_proba(input_df)[0][1]
        pred     = r["final_pipe"].predict(input_df)[0]

        st.markdown("---")
        badge_cls = "pred-positive" if pred == 1 else "pred-negative"
        badge_lbl = "⚠️  HIGH RISK" if pred == 1 else "✅  LOW RISK"
        st.markdown(
            f"<div style='text-align:center;margin:20px 0;'>"
            f"<div class='{badge_cls}'>{badge_lbl}</div>"
            f"</div>", unsafe_allow_html=True
        )

        st.markdown(
            f"<div class='metric-card' style='max-width:320px;margin:auto;'>"
            f"<div class='metric-label'>Risk Probability</div>"
            f"<div class='metric-value'>{prob:.1%}</div>"
            f"<div class='metric-sub'>Tuned XGBoost · External AUC {r['ext_auc']:.4f}</div>"
            f"</div>", unsafe_allow_html=True
        )

        # Mini gauge bar
        st.markdown("<br>", unsafe_allow_html=True)
        bar_color = "#f85149" if prob > 0.5 else "#3fb950"
        st.markdown(
            f"<div style='background:#21262d;border-radius:8px;height:14px;overflow:hidden;'>"
            f"<div style='background:{bar_color};width:{prob*100:.1f}%;height:100%;'></div>"
            f"</div>"
            f"<p style='text-align:right;font-size:0.75rem;color:#8b949e;margin:4px 0 0;'>"
            f"{prob*100:.1f}% risk</p>",
            unsafe_allow_html=True
        )

# ── Tab 6: Data ──────────────────────────────────────────
with tabs[6]:
    st.markdown("<div class='section-header'>Dataset Overview</div>",
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Features", df.shape[1] - 1)
    c3.metric("Class Balance", f"{r['y'].mean():.1%} positive")

    st.dataframe(df.head(50), use_container_width=True, hide_index=True)

    st.markdown("<div class='section-header'>Descriptive Statistics</div>",
                unsafe_allow_html=True)
    st.dataframe(df.describe().T.round(3), use_container_width=True)

    st.markdown("<div class='section-header'>Correlation Heatmap</div>",
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    mask = np.triu(np.ones_like(df.corr(), dtype=bool), k=1)
    sns.heatmap(df.corr(), mask=~mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.4,
                ax=ax, annot_kws={"size": 7},
                cbar_kws={"shrink": 0.7})
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    st.image(fig_to_st(fig))

    # Download CSV
    csv = r["results_df"].to_csv(index=False).encode()
    st.download_button(
        "⬇️  Download CV Results Table",
        data=csv,
        file_name="Table_A_Results.csv",
        mime="text/csv"
    )
