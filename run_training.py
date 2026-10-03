"""
run_training.py
---------------
Complete end-to-end training pipeline for FloodAI.

Usage (from FloodAI/ directory):
    python run_training.py

Steps executed:
    1. Load dataset
    2. Validate dataset
    3. Clean data
    4. Exploratory Data Analysis
    5. Feature engineering
    6. Train models (LR, RF, XGB)
    7. Evaluate models
    8. Select & save best model + pipeline
    9. Save evaluation metrics
   10. Compute & save SHAP explanations
   11. Generate all figures
   12. Generate dataset report
"""

import json
import logging
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_RAW      = os.path.join(PROJECT_ROOT, "data", "raw",       "flood_risk_dataset_india.csv")
DATA_PROC     = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR   = os.path.join(PROJECT_ROOT, "outputs", "figures")
REPORTS_DIR   = os.path.join(PROJECT_ROOT, "outputs", "reports")

for d in [DATA_PROC, MODELS_DIR, FIGURES_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(PROJECT_ROOT, "training.log"), mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("FloodAI — Training Pipeline  START")
    logger.info("=" * 60)

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    logger.info("[1/10] Loading dataset …")
    from src.preprocessing import load_raw_data, TARGET_COL
    df_raw = load_raw_data(DATA_RAW)
    logger.info("Dataset shape: %d rows × %d columns", *df_raw.shape)

    # ── 2. Validate dataset ───────────────────────────────────────────────────
    logger.info("[2/10] Validating dataset …")
    from src.preprocessing import validate_dataset, save_report
    report = validate_dataset(df_raw)
    save_report(report, os.path.join(REPORTS_DIR, "dataset_validation.json"))
    logger.info("Missing values: %d | Duplicates: %d",
                report["total_missing"], report["duplicate_rows"])

    # ── 3. Clean data ─────────────────────────────────────────────────────────
    logger.info("[3/10] Cleaning data …")
    from src.preprocessing import clean_data
    df_clean = clean_data(df_raw)

    # Save processed data
    processed_path = os.path.join(DATA_PROC, "flood_risk_processed.csv")
    df_clean.to_csv(processed_path, index=False)
    logger.info("Cleaned data saved: %s", processed_path)

    # ── 4. EDA ────────────────────────────────────────────────────────────────
    logger.info("[4/10] Running Exploratory Data Analysis …")
    from src.eda import run_full_eda
    run_full_eda(df_clean, FIGURES_DIR)

    # ── 5. Feature engineering ────────────────────────────────────────────────
    logger.info("[5/10] Feature engineering …")
    from src.feature_engineering import engineer_features
    df_engineered = engineer_features(df_clean)
    eng_path = os.path.join(DATA_PROC, "flood_risk_engineered.csv")
    df_engineered.to_csv(eng_path, index=False)
    logger.info("Engineered data saved: %s  (shape: %d × %d)", eng_path, *df_engineered.shape)

    # ── 6-9. Train, evaluate, save models ────────────────────────────────────
    logger.info("[6/10] Training models …")
    from src.train import train_models
    results = train_models(
        df=df_engineered,
        target_col=TARGET_COL,
        models_dir=MODELS_DIR,
        figures_dir=FIGURES_DIR,
        reports_dir=REPORTS_DIR,
    )

    logger.info("[7/10] Best model: %s  AUC=%.4f",
                results["best_model_name"],
                max(m["roc_auc"] for m in results["metrics"]))

    # ── 10. SHAP explanations ─────────────────────────────────────────────────
    logger.info("[8/10] Computing SHAP explanations …")
    try:
        from src.explainability import (
            compute_shap_values,
            save_shap_summary_plot,
            save_shap_bar_plot,
        )
        best_pipeline = joblib.load(results["best_model_path"])
        X_test = results["X_test"]
        shap_vals, X_transformed, feat_names = compute_shap_values(
            best_pipeline, X_test, max_samples=300
        )

        save_shap_summary_plot(
            shap_vals, X_transformed, feat_names,
            os.path.join(FIGURES_DIR, "shap_summary.png"),
        )
        save_shap_bar_plot(
            shap_vals, feat_names,
            os.path.join(FIGURES_DIR, "shap_importance.png"),
        )

        # Save feature names for the Streamlit app
        shap_meta = {
            "feature_names": feat_names,
            "best_model": results["best_model_name"],
            "num_features": results["num_features"],
            "cat_features": results["cat_features"],
        }
        with open(os.path.join(MODELS_DIR, "model_meta.json"), "w") as fh:
            json.dump(shap_meta, fh, indent=2)
        logger.info("SHAP plots saved.")

    except Exception as exc:
        logger.warning("SHAP computation skipped: %s", exc)

    # ── 11. Save metrics for Streamlit app ────────────────────────────────────
    logger.info("[9/10] Saving metrics …")
    metrics_out = []
    for m in results["metrics"]:
        metrics_out.append({k: v for k, v in m.items()
                             if k not in ("y_pred", "y_prob")})

    # Also save test data with predictions for the risk map
    logger.info("[10/10] Saving risk map data …")
    try:
        from src.prediction import predict_batch
        best_pipeline = joblib.load(results["best_model_path"])
        X_test_copy = results["X_test"].copy()
        y_test = results["y_test"]
        X_test_copy[TARGET_COL] = y_test.values
        preds_df = predict_batch(best_pipeline, results["X_test"])
        preds_df[TARGET_COL] = y_test.values

        # Merge with lat/lon if available
        map_df = preds_df[["Latitude", "Longitude",
                            "Flood_Probability", "Risk_Category",
                            "Flood_Prediction", TARGET_COL]].copy()
        map_path = os.path.join(DATA_PROC, "test_predictions.csv")
        map_df.to_csv(map_path, index=False)
        logger.info("Risk map data saved: %s", map_path)
    except Exception as exc:
        logger.warning("Risk map data generation skipped: %s", exc)

    logger.info("=" * 60)
    logger.info("FloodAI — Training Pipeline  COMPLETE")
    logger.info("Models saved in : %s", MODELS_DIR)
    logger.info("Figures saved in: %s", FIGURES_DIR)
    logger.info("Reports saved in: %s", REPORTS_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
