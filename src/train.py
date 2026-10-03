"""
train.py
--------
Trains Logistic Regression, Random Forest, and XGBoost models.
Evaluates each model and saves artifacts to models/ and outputs/.
"""

import json
import logging
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocessing import build_preprocessor, get_feature_lists

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_model_pipeline(preprocessor, model):
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])


def _evaluate(name, pipeline, X_test, y_test):
    """Return a metrics dict for a fitted pipeline."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    return {
        "model": name,
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_test, y_prob)),
        "y_pred":    y_pred.tolist(),
        "y_prob":    y_prob.tolist(),
    }


def _save_confusion_matrix(y_test, y_pred, name, out_dir):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Flood", "Flood"],
                yticklabels=["No Flood", "Flood"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}")
    fig.tight_layout()
    path = os.path.join(out_dir, f"confusion_matrix_{name.replace(' ', '_')}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix: %s", path)


def _save_roc_curve(metrics_list, y_test, pipelines, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for m, (name, pip) in zip(metrics_list, pipelines.items()):
        y_prob = pip.predict_proba(y_test.index.map(lambda i: None))
        fpr, tpr, _ = roc_curve(y_test, m["y_prob"])
        ax.plot(fpr, tpr, label=f"{name}  AUC={m['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "roc_curves_all.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved ROC curves: %s", path)


def _save_pr_curve(metrics_list, y_test, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in metrics_list:
        prec, rec, _ = precision_recall_curve(y_test, m["y_prob"])
        ax.plot(rec, prec, label=f"{m['model']}  F1={m['f1']:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Models")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "pr_curves_all.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved PR curves: %s", path)


def _save_model_comparison_chart(metrics_list, out_dir):
    names = [m["model"] for m in metrics_list]
    metrics_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(names))
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, key in enumerate(metrics_keys):
        vals = [m[key] for m in metrics_list]
        ax.bar(x + i * width, vals, width, label=key.capitalize())

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "model_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved model comparison chart: %s", path)


# ── main training function ────────────────────────────────────────────────────

def train_models(
    df: pd.DataFrame,
    target_col: str,
    models_dir: str,
    figures_dir: str,
    reports_dir: str,
) -> dict:
    """
    Full training workflow:
    1. Split data
    2. Build preprocessor
    3. Train LR, RF, XGB
    4. Evaluate
    5. Save best model + all models
    6. Generate figures
    7. Return metrics dict

    Returns
    -------
    dict with keys: 'metrics', 'best_model_name', 'best_model_path',
                    'pipeline_path', 'X_test', 'y_test', 'feature_names'
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(
        "Train/test split: %d / %d  (positives: %d / %d)",
        len(X_train), len(X_test),
        y_train.sum(), y_test.sum(),
    )

    num_features, cat_features = get_feature_lists(X_train)
    preprocessor = build_preprocessor(num_features, cat_features)

    # ── model definitions ────────────────────────────────────────────────────
    lr_model = LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
    )
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=RANDOM_STATE,
        class_weight="balanced", n_jobs=-1
    )
    xgb_model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=RANDOM_STATE, eval_metric="logloss",
        use_label_encoder=False, n_jobs=-1,
        scale_pos_weight=float((y_train == 0).sum()) / float((y_train == 1).sum()),
    )

    model_defs = {
        "Logistic Regression": lr_model,
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
    }

    fitted_pipelines = {}
    metrics_list = []

    for name, model in model_defs.items():
        logger.info("Training: %s …", name)
        pipeline = _make_model_pipeline(build_preprocessor(num_features, cat_features), model)
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        m = _evaluate(name, pipeline, X_test, y_test)
        metrics_list.append(m)
        logger.info(
            "  %s  Acc=%.4f  F1=%.4f  AUC=%.4f",
            name, m["accuracy"], m["f1"], m["roc_auc"],
        )

        _save_confusion_matrix(y_test, m["y_pred"], name, figures_dir)

        # Save individual model
        model_path = os.path.join(models_dir, f"{name.replace(' ', '_')}_pipeline.pkl")
        joblib.dump(pipeline, model_path)
        logger.info("  Saved pipeline: %s", model_path)

    # ── select best model by ROC-AUC ─────────────────────────────────────────
    best = max(metrics_list, key=lambda m: m["roc_auc"])
    best_name = best["model"]
    best_pipeline = fitted_pipelines[best_name]
    logger.info("Best model: %s  (AUC=%.4f)", best_name, best["roc_auc"])

    best_path = os.path.join(models_dir, "best_model_pipeline.pkl")
    joblib.dump(best_pipeline, best_path)
    logger.info("Best model saved: %s", best_path)

    # ── generate figures ──────────────────────────────────────────────────────
    _save_pr_curve(metrics_list, y_test, figures_dir)
    _save_model_comparison_chart(metrics_list, figures_dir)

    # ROC curves (use pre-computed probs)
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in metrics_list:
        fpr, tpr, _ = roc_curve(y_test, m["y_prob"])
        ax.plot(fpr, tpr, label=f"{m['model']}  AUC={m['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend()
    fig.tight_layout()
    roc_path = os.path.join(figures_dir, "roc_curves_all.png")
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)

    # ── save metrics report ───────────────────────────────────────────────────
    # Strip raw arrays before serialising
    clean_metrics = []
    for m in metrics_list:
        cm = {k: v for k, v in m.items() if k not in ("y_pred", "y_prob")}
        clean_metrics.append(cm)

    report = {
        "best_model": best_name,
        "models": clean_metrics,
    }
    report_path = os.path.join(reports_dir, "model_metrics.json")
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Metrics report saved: %s", report_path)

    # ── get feature names from the best pipeline ──────────────────────────────
    try:
        ct = best_pipeline.named_steps["preprocessor"]
        ohe = ct.named_transformers_["cat"].named_steps["encoder"]
        cat_feat_names = ohe.get_feature_names_out(cat_features).tolist()
        feature_names = num_features + cat_feat_names
    except Exception:
        feature_names = num_features + cat_features

    return {
        "metrics": metrics_list,
        "best_model_name": best_name,
        "best_model_path": best_path,
        "pipeline_path": best_path,
        "feature_names": feature_names,
        "num_features": num_features,
        "cat_features": cat_features,
        "X_test": X_test,
        "y_test": y_test,
        "fitted_pipelines": fitted_pipelines,
    }
