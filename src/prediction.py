"""
prediction.py
-------------
Flood-risk scoring utilities used by both the training pipeline
and the Streamlit application.
"""

import logging
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Application-defined visualisation thresholds (NOT official government standards)
RISK_THRESHOLDS = {
    "Low":       (0.00, 0.25),
    "Moderate":  (0.25, 0.50),
    "High":      (0.50, 0.75),
    "Very High": (0.75, 1.01),
}


def probability_to_category(prob: float) -> str:
    """Map a flood probability [0, 1] to a risk category string."""
    for category, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= prob < hi:
            return category
    return "Very High"


def predict_single(pipeline, input_dict: dict) -> dict:
    """
    Make a flood-risk prediction for a single observation.

    Parameters
    ----------
    pipeline : sklearn Pipeline (preprocessor + classifier)
    input_dict : dict mapping feature names to values

    Returns
    -------
    dict with keys:
        flood_probability, risk_percentage, risk_category,
        flood_prediction (0 or 1), flood_label ("Flood" / "No Flood")
    """
    df = pd.DataFrame([input_dict])
    prob = float(pipeline.predict_proba(df)[0, 1])
    pred = int(pipeline.predict(df)[0])
    category = probability_to_category(prob)

    return {
        "flood_probability": prob,
        "risk_percentage": round(prob * 100, 2),
        "risk_category": category,
        "flood_prediction": pred,
        "flood_label": "Flood" if pred == 1 else "No Flood",
    }


def predict_batch(pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """
    Add prediction columns to a copy of df (must NOT contain the target column).

    Returns df with added columns:
        Flood_Probability, Risk_Percentage, Risk_Category, Flood_Prediction
    """
    out = df.copy()
    probs = pipeline.predict_proba(df)[:, 1]
    preds = pipeline.predict(df)

    out["Flood_Probability"] = probs
    out["Risk_Percentage"]   = (probs * 100).round(2)
    out["Risk_Category"]     = [probability_to_category(p) for p in probs]
    out["Flood_Prediction"]  = preds
    return out


def get_feature_importance(pipeline, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importances from the classifier inside the pipeline.
    Works for Random Forest (feature_importances_) and XGBoost.
    Falls back to coefficient magnitudes for Logistic Regression.

    Returns a DataFrame sorted by importance descending.
    """
    clf = pipeline.named_steps["classifier"]

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        logger.warning("Classifier has no feature importance attribute.")
        return pd.DataFrame(columns=["feature", "importance"])

    n = min(len(importances), len(feature_names))
    fi = pd.DataFrame(
        {"feature": feature_names[:n], "importance": importances[:n]}
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    return fi
