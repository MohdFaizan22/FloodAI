"""
preprocessing.py
----------------
Data loading, validation, cleaning, and sklearn Pipeline construction
for the FloodAI project.

Dataset: flood_risk_dataset_india.csv  (synthetic, 10 000 rows, 14 columns)
Target : Flood Occurred  (binary 0/1)
"""

import os
import json
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# ── Column definitions (derived from actual CSV header) ─────────────────────
TARGET_COL = "Flood Occurred"

NUMERICAL_FEATURES = [
    "Latitude",
    "Longitude",
    "Rainfall (mm)",
    "Temperature (°C)",
    "Humidity (%)",
    "River Discharge (m³/s)",
    "Water Level (m)",
    "Elevation (m)",
    "Population Density",
    "Infrastructure",
    "Historical Floods",
]

CATEGORICAL_FEATURES = ["Land Cover", "Soil Type"]

# Engineered features added by feature_engineering.py
ENGINEERED_NUMERICAL = [
    "Rainfall_WaterLevel_Interaction",
    "Rainfall_Discharge_Interaction",
    "Rainfall_Intensity",
    "WaterLevel_Severity",
    "Discharge_Risk",
    "Low_Elevation_Flag",
    "Population_Exposure",
]


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV and return a DataFrame."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info("Loaded dataset: %d rows × %d columns", *df.shape)
    return df


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Inspect the DataFrame and return a structured validation report.
    Does NOT modify the DataFrame.
    """
    report = {}
    report["n_rows"] = int(df.shape[0])
    report["n_cols"] = int(df.shape[1])
    report["columns"] = list(df.columns)
    report["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
    report["missing_values"] = df.isnull().sum().to_dict()
    report["total_missing"] = int(df.isnull().sum().sum())
    report["duplicate_rows"] = int(df.duplicated().sum())

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    report["numerical_ranges"] = {
        c: {"min": float(df[c].min()), "max": float(df[c].max()), "mean": float(df[c].mean())}
        for c in num_cols
    }

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    report["categorical_unique"] = {c: df[c].unique().tolist() for c in cat_cols}

    if TARGET_COL in df.columns:
        vc = df[TARGET_COL].value_counts().to_dict()
        report["target_distribution"] = {str(k): int(v) for k, v in vc.items()}
        total = df.shape[0]
        report["class_balance"] = {
            str(k): f"{100*v/total:.2f}%" for k, v in vc.items()
        }

    logger.info("Validation report generated.")
    return report


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic cleaning:
    - Drop exact duplicate rows
    - Coerce numeric columns
    - Clip obvious outliers using IQR × 3 fence (very conservative)
    Returns a cleaned copy.
    """
    df = df.copy()

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        logger.info("Removed %d duplicate rows.", before - after)

    # Ensure numeric columns are numeric
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clip extreme outliers with a 3×IQR fence (keeps 99.9 %+ of data)
    for col in [c for c in NUMERICAL_FEATURES if c in df.columns]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        df[col] = df[col].clip(lower, upper)

    logger.info("Data cleaned. Final shape: %d × %d", *df.shape)
    return df


def build_preprocessor(num_features: list, cat_features: list) -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer:
    - Numerical: median imputation + standard scaling
    - Categorical: most-frequent imputation + one-hot encoding
    """
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_features),
            ("cat", cat_pipeline, cat_features),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_lists(df: pd.DataFrame) -> tuple[list, list]:
    """
    Return (numerical_features, categorical_features) present in df,
    excluding the target column.
    """
    all_num = NUMERICAL_FEATURES + ENGINEERED_NUMERICAL
    num = [c for c in all_num if c in df.columns and c != TARGET_COL]
    cat = [c for c in CATEGORICAL_FEATURES if c in df.columns and c != TARGET_COL]
    return num, cat


def save_report(report: dict, output_path: str) -> None:
    """Persist a dictionary as a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    logger.info("Report saved to %s", output_path)
