"""
explainability.py
-----------------
SHAP-based global and individual explanation utilities for FloodAI.

IMPORTANT DISCLAIMER
--------------------
Feature importances and SHAP values describe statistical associations
learned by the model. They do NOT establish causal relationships between
features and flood occurrence. Do not interpret these results as causal
proof of any physical mechanism.
"""

import logging
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def _get_shap_explainer(pipeline, X_background: pd.DataFrame):
    """
    Create a SHAP explainer appropriate for the classifier in pipeline.
    Uses TreeExplainer for tree-based models, LinearExplainer for LR.
    """
    import shap

    clf = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X_background)

    clf_type = type(clf).__name__
    if clf_type in ("RandomForestClassifier", "XGBClassifier",
                    "GradientBoostingClassifier", "ExtraTreesClassifier"):
        explainer = shap.TreeExplainer(clf)
    else:
        explainer = shap.LinearExplainer(clf, X_transformed,
                                         feature_perturbation="interventional")
    return explainer, X_transformed


def compute_shap_values(pipeline, X_sample: pd.DataFrame, max_samples: int = 300):
    """
    Compute SHAP values for up to max_samples rows.

    Returns
    -------
    shap_values : np.ndarray  (n_samples × n_features)
    X_transformed : np.ndarray
    feature_names : list[str]
    """
    import shap

    # Sample for speed
    if len(X_sample) > max_samples:
        X_sample = X_sample.sample(max_samples, random_state=42)

    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]
    X_transformed = preprocessor.transform(X_sample)

    clf_type = type(clf).__name__
    if clf_type in ("RandomForestClassifier", "XGBClassifier",
                    "GradientBoostingClassifier", "ExtraTreesClassifier"):
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_transformed)
        # For binary classifiers TreeExplainer returns list of 2 arrays
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            shap_vals = shap_vals[1]
    else:
        background = shap.maskers.Independent(X_transformed, max_samples=100)
        explainer = shap.LinearExplainer(clf, background)
        shap_vals = explainer.shap_values(X_transformed)

    # Resolve feature names from ColumnTransformer
    try:
        ct = preprocessor
        num_names = ct.transformers_[0][2]
        ohe = ct.named_transformers_["cat"].named_steps["encoder"]
        cat_names = ohe.get_feature_names_out(ct.transformers_[1][2]).tolist()
        feature_names = list(num_names) + cat_names
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    return shap_vals, X_transformed, feature_names


def save_shap_summary_plot(shap_values, X_transformed, feature_names, output_path: str):
    """Save a SHAP beeswarm summary plot."""
    import shap

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Keep top 20 features for readability
    top_n = 20
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-top_n:][::-1]
    sv_top = shap_values[:, top_idx]
    xt_top = X_transformed[:, top_idx]
    fn_top = [feature_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(sv_top, xt_top, feature_names=fn_top, show=False,
                      plot_type="dot", max_display=top_n)
    plt.title("SHAP Summary — Global Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP summary plot saved: %s", output_path)


def save_shap_bar_plot(shap_values, feature_names, output_path: str):
    """Save a SHAP mean |value| bar chart."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_n = min(20, len(feature_names))
    top_idx = np.argsort(mean_abs)[-top_n:][::-1]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        [feature_names[i] for i in top_idx][::-1],
        mean_abs[top_idx][::-1],
        color="#3b82d4",
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance (SHAP)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("SHAP bar plot saved: %s", output_path)


def explain_single_prediction(pipeline, row: pd.DataFrame, feature_names: list) -> dict:
    """
    Compute SHAP values for a single row and return a tidy contribution dict.

    Returns
    -------
    dict: {feature_name: shap_value, ...}  sorted by |shap_value| descending
    """
    import shap

    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]
    X_t = preprocessor.transform(row)

    clf_type = type(clf).__name__
    if clf_type in ("RandomForestClassifier", "XGBClassifier",
                    "GradientBoostingClassifier"):
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_t)
        if isinstance(sv, list) and len(sv) == 2:
            sv = sv[1]
    else:
        background = shap.maskers.Independent(X_t, max_samples=50)
        explainer = shap.LinearExplainer(clf, background)
        sv = explainer.shap_values(X_t)

    sv = sv[0] if sv.ndim > 1 else sv
    n = min(len(sv), len(feature_names))
    contribs = {feature_names[i]: float(sv[i]) for i in range(n)}
    return dict(sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True))
