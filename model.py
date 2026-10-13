"""LEDGERLENS Anomaly Detection Model and Data Processing.

This module provides data loading, sanitization, and baseline feature extraction.
"""
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

DATASET_PATH = Path(__file__).resolve().parent / "sample_projects.csv"

# Baseline 6 indicators from original prototype
BASELINE_FEATURES = [
    "cost_overrun",
    "fund_utilisation",
    "funds_released_ratio",
    "time_elapsed",
    "progress_vs_time",
    "spend_vs_progress",
]

FEATURES = BASELINE_FEATURES


def load_data(filepath: Optional[Union[str, Path]] = None, sanitize: bool = True) -> pd.DataFrame:
    """Load, validate, and sanitize project data with defensive denominator handling.

    Parameters
    ----------
    filepath : str or Path, optional
        Path to the project CSV. Defaults to sample_projects.csv.
    sanitize : bool, default=True
        Whether to explicitly sanitize missing, zero, and extreme values.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns and sanitized metrics.
    """
    path = Path(filepath) if filepath else DATASET_PATH
    data = pd.read_csv(path)

    # Standardize column naming
    rename = {
        "project_id": "id",
        "project_name": "name",
        "estimated_cost": "est_cost",
        "funds_utilised": "funds_used",
        "planned_days": "planned_days",
        "elapsed_days": "elapsed_days",
        "progress": "progress_pct",
    }
    data = data.rename(columns=rename)

    # 1. Handle missing string/category identifiers
    if "id" not in data.columns:
        data["id"] = [f"PRJ-{i + 2000}" for i in range(len(data))]
    else:
        data["id"] = data["id"].fillna(pd.Series([f"PRJ-{i + 2000}" for i in range(len(data))]))

    if "name" not in data.columns:
        data["name"] = data["id"]
    else:
        data["name"] = data["name"].fillna(data["id"])

    if "sector" not in data.columns:
        data["sector"] = "General"
    else:
        data["sector"] = data["sector"].fillna("General")

    if "district" not in data.columns:
        data["district"] = "Unknown"
    else:
        data["district"] = data["district"].fillna("Unknown")

    # 2. Convert and validate numeric attributes
    numeric_cols = [
        "est_cost",
        "actual_cost",
        "funds_released",
        "funds_used",
        "planned_days",
        "elapsed_days",
        "progress_pct",
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        else:
            data[col] = np.nan

    if sanitize:
        # Handle non-positive or missing estimated costs
        est_valid = data["est_cost"].dropna()
        est_median = est_valid[est_valid > 0].median() if len(est_valid[est_valid > 0]) > 0 else 1.0
        data["est_cost"] = data["est_cost"].fillna(est_median).clip(lower=0.0)

        # Actual costs and disbursements
        data["actual_cost"] = data["actual_cost"].fillna(data["est_cost"]).clip(lower=0.0)
        data["funds_released"] = data["funds_released"].fillna(0.0).clip(lower=0.0)
        data["funds_used"] = data["funds_used"].fillna(0.0).clip(lower=0.0)

        # Planned and elapsed project durations
        planned_valid = data["planned_days"].dropna()
        planned_median = planned_valid[planned_valid > 0].median() if len(planned_valid[planned_valid > 0]) > 0 else 365.0
        data["planned_days"] = data["planned_days"].fillna(planned_median).clip(lower=1.0)
        data["elapsed_days"] = data["elapsed_days"].fillna(0.0).clip(lower=0.0)

        # Normalize progress scale: if [0, 1.5], scale to percentage [0, 100]
        prog = data["progress_pct"].fillna(0.0)
        if prog.max(skipna=True) <= 1.5:
            prog = prog * 100.0
        data["progress_pct"] = prog.clip(0.0, 100.0)

    # Safe denominator series avoiding zero-division
    safe_est_cost = np.where(data["est_cost"] > 0, data["est_cost"], np.nan)
    safe_funds_released = np.where(data["funds_released"] > 0, data["funds_released"], np.nan)
    safe_planned_days = np.where(data["planned_days"] > 0, data["planned_days"], np.nan)

    # Baseline indicators with safe division
    data["cost_overrun"] = (data["actual_cost"] - data["est_cost"]) / safe_est_cost
    data["cost_overrun"] = data["cost_overrun"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    data["fund_utilisation"] = data["funds_used"] / safe_funds_released
    data["fund_utilisation"] = (
        data["fund_utilisation"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
    )

    data["funds_released_ratio"] = data["funds_released"] / safe_est_cost
    data["funds_released_ratio"] = data["funds_released_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    data["time_elapsed"] = data["elapsed_days"] / safe_planned_days
    data["time_elapsed"] = data["time_elapsed"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    data["progress_vs_time"] = data["progress_pct"] / 100.0 - data["time_elapsed"].clip(upper=1.0)
    data["spend_vs_progress"] = data["fund_utilisation"] - data["progress_pct"] / 100.0

    return data.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def run_anomaly_detection(contamination: float = 0.12, random_state: int = 42) -> Dict[str, object]:
    """Execute baseline Isolation Forest anomaly detection."""
    data = load_data()
    scaler = StandardScaler()
    X = scaler.fit_transform(data[FEATURES])
    model = IsolationForest(n_estimators=250, contamination=contamination, random_state=random_state)
    data["anomaly"] = model.fit_predict(X)
    data["anomaly_score"] = model.decision_function(X)
    data["anomaly_strength"] = (-data["anomaly_score"]).clip(lower=0.0)
    return {"data": data, "model": model, "scaler": scaler, "features": FEATURES}


def explain_project(row: pd.Series, data: pd.DataFrame) -> List[str]:
    """Generate rule-based reasons for baseline flagged projects."""
    peers = data[data["sector"] == row["sector"]]
    reasons = []
    peer_cost = peers["cost_overrun"].median()
    peer_util = peers["fund_utilisation"].median()
    peer_gap = peers["spend_vs_progress"].median()
    if row["cost_overrun"] > peer_cost + 0.15:
        reasons.append(f"Cost overrun is {row['cost_overrun']:.0%}, above the sector peer median of {peer_cost:.0%}.")
    if row["fund_utilisation"] > peer_util + 0.15:
        reasons.append(f"Fund utilisation is {row['fund_utilisation']:.0%}, above the sector peer median of {peer_util:.0%}.")
    if row["spend_vs_progress"] > peer_gap + 0.20:
        reasons.append("Spending is running ahead of reported physical progress.")
    if row["progress_vs_time"] < -0.20:
        reasons.append("Physical progress is materially behind the elapsed timeline.")
    if not reasons and row.get("anomaly", 1) == -1:
        reasons.append("The combination of project metrics is unusual relative to the overall portfolio.")
    return reasons
