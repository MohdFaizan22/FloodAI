"""
feature_engineering.py
-----------------------
Creates domain-informed features supported by the actual dataset columns.

All engineered features are computed before the sklearn preprocessor runs,
so they participate in scaling and imputation like any raw numerical column.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to a copy of df.
    All feature names are registered in preprocessing.ENGINEERED_NUMERICAL.

    Features created
    ----------------
    Rainfall_WaterLevel_Interaction
        Product of normalised Rainfall and Water Level.
        High rainfall combined with already-elevated water levels is a
        strong flood precursor.

    Rainfall_Discharge_Interaction
        Product of normalised Rainfall and River Discharge.
        Heavy rain on a river already running at high discharge raises
        flood probability.

    Rainfall_Intensity
        Rainfall divided by (1 + Humidity/100), a crude proxy for
        effective rainfall intensity after atmosphere saturation is
        considered.

    WaterLevel_Severity
        Water Level divided by (1 + Elevation/10 000), weighting high
        water in low-elevation areas as more severe.

    Discharge_Risk
        Log1p-transformed River Discharge, capturing diminishing returns
        in risk growth at very high discharge values.

    Low_Elevation_Flag
        Binary flag: 1 if Elevation < 500 m (low-lying, flood-prone zone).

    Population_Exposure
        Population Density × (1 − min-max normalised Elevation).
        Represents people in low-elevation, densely populated areas.
    """
    df = df.copy()

    # Safe column access helpers
    def _col(name):
        return df[name] if name in df.columns else pd.Series(0, index=df.index)

    rainfall   = _col("Rainfall (mm)")
    water_lvl  = _col("Water Level (m)")
    discharge  = _col("River Discharge (m³/s)")
    humidity   = _col("Humidity (%)")
    elevation  = _col("Elevation (m)")
    pop_den    = _col("Population Density")

    # Min-max normalise within the current split to avoid leakage;
    # the actual scaling is handled later by StandardScaler.
    def _norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0

    df["Rainfall_WaterLevel_Interaction"] = _norm(rainfall) * _norm(water_lvl)
    df["Rainfall_Discharge_Interaction"]  = _norm(rainfall) * _norm(discharge)
    df["Rainfall_Intensity"]              = rainfall / (1 + humidity / 100)
    df["WaterLevel_Severity"]             = water_lvl / (1 + elevation / 10_000)
    df["Discharge_Risk"]                  = np.log1p(discharge)
    df["Low_Elevation_Flag"]              = (elevation < 500).astype(int)
    df["Population_Exposure"]             = pop_den * (1 - _norm(elevation))

    logger.info(
        "Feature engineering complete. Added 7 features. New shape: %d × %d",
        *df.shape,
    )
    return df
