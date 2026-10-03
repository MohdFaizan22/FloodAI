"""
eda.py
------
Exploratory Data Analysis visualisations for the FloodAI project.
All figures are saved to the outputs/figures/ directory.
"""

import logging
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

TARGET = "Flood Occurred"
PALETTE = {0: "#3b82d4", 1: "#e74c3c"}


def _savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)


def plot_target_distribution(df, out_dir):
    vc = df[TARGET].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["No Flood (0)", "Flood (1)"], [vc.get(0, 0), vc.get(1, 0)],
           color=["#3b82d4", "#e74c3c"])
    ax.set_title("Flood vs Non-Flood Distribution")
    ax.set_ylabel("Count")
    for i, v in enumerate([vc.get(0, 0), vc.get(1, 0)]):
        ax.text(i, v + 20, str(v), ha="center", fontsize=10)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "target_distribution.png"))


def plot_numerical_distributions(df, out_dir):
    num_cols = ["Rainfall (mm)", "Temperature (°C)", "Humidity (%)",
                "River Discharge (m³/s)", "Water Level (m)", "Elevation (m)",
                "Population Density"]
    num_cols = [c for c in num_cols if c in df.columns]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        for flood_val in [0, 1]:
            subset = df[df[TARGET] == flood_val][col].dropna()
            axes[i].hist(subset, bins=40, alpha=0.6,
                         color=PALETTE[flood_val],
                         label="Flood" if flood_val else "No Flood")
        axes[i].set_title(col, fontsize=9)
        axes[i].legend(fontsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Numerical Feature Distributions by Flood Occurrence")
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "numerical_distributions.png"))


def plot_correlation_matrix(df, out_dir):
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, annot_kws={"size": 7})
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "correlation_matrix.png"))


def plot_rainfall_vs_waterlevel(df, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for val in [0, 1]:
        sub = df[df[TARGET] == val]
        ax.scatter(sub["Rainfall (mm)"], sub["Water Level (m)"],
                   alpha=0.3, s=8,
                   color=PALETTE[val],
                   label="Flood" if val else "No Flood")
    ax.set_xlabel("Rainfall (mm)")
    ax.set_ylabel("Water Level (m)")
    ax.set_title("Rainfall vs Water Level")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "rainfall_vs_waterlevel.png"))


def plot_rainfall_vs_discharge(df, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for val in [0, 1]:
        sub = df[df[TARGET] == val]
        ax.scatter(sub["Rainfall (mm)"], sub["River Discharge (m³/s)"],
                   alpha=0.3, s=8,
                   color=PALETTE[val],
                   label="Flood" if val else "No Flood")
    ax.set_xlabel("Rainfall (mm)")
    ax.set_ylabel("River Discharge (m³/s)")
    ax.set_title("Rainfall vs River Discharge")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "rainfall_vs_discharge.png"))


def plot_flood_by_land_cover(df, out_dir):
    if "Land Cover" not in df.columns:
        return
    ct = df.groupby(["Land Cover", TARGET]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ct.plot(kind="bar", ax=ax, color=["#3b82d4", "#e74c3c"])
    ax.set_title("Flood Occurrence by Land Cover")
    ax.set_xlabel("Land Cover")
    ax.set_ylabel("Count")
    ax.legend(["No Flood", "Flood"])
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "flood_by_land_cover.png"))


def plot_flood_by_soil_type(df, out_dir):
    if "Soil Type" not in df.columns:
        return
    ct = df.groupby(["Soil Type", TARGET]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ct.plot(kind="bar", ax=ax, color=["#3b82d4", "#e74c3c"])
    ax.set_title("Flood Occurrence by Soil Type")
    ax.set_xlabel("Soil Type")
    ax.set_ylabel("Count")
    ax.legend(["No Flood", "Flood"])
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "flood_by_soil_type.png"))


def plot_historical_floods_vs_current(df, out_dir):
    if "Historical Floods" not in df.columns:
        return
    ct = df.groupby(["Historical Floods", TARGET]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(5, 4))
    ct.plot(kind="bar", ax=ax, color=["#3b82d4", "#e74c3c"])
    ax.set_title("Historical Floods vs Current Flood Occurrence")
    ax.set_xlabel("Historical Floods (0=No, 1=Yes)")
    ax.set_ylabel("Count")
    ax.legend(["No Flood", "Flood"])
    plt.xticks(rotation=0)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "historical_vs_current_flood.png"))


def plot_geographic_scatter(df, out_dir):
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    for val in [0, 1]:
        sub = df[df[TARGET] == val]
        ax.scatter(sub["Longitude"], sub["Latitude"], alpha=0.2, s=5,
                   color=PALETTE[val],
                   label="Flood" if val else "No Flood")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Geographic Distribution of Flood Occurrences")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "geographic_scatter.png"))


def run_full_eda(df, out_dir):
    """Generate all EDA plots."""
    logger.info("Running EDA …")
    plot_target_distribution(df, out_dir)
    plot_numerical_distributions(df, out_dir)
    plot_correlation_matrix(df, out_dir)
    plot_rainfall_vs_waterlevel(df, out_dir)
    plot_rainfall_vs_discharge(df, out_dir)
    plot_flood_by_land_cover(df, out_dir)
    plot_flood_by_soil_type(df, out_dir)
    plot_historical_floods_vs_current(df, out_dir)
    plot_geographic_scatter(df, out_dir)
    logger.info("EDA complete. Figures saved to %s", out_dir)
