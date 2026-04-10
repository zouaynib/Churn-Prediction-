"""Feature engineering for Telco Churn dataset.

Creates domain-specific features BEFORE the sklearn ColumnTransformer.
All transforms are pure column operations on pandas DataFrames,
so they can be applied identically to train / val / test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to a copy of *df* and return it."""
    df = df.copy()

    # ── Charge-based features ────────────────────────────────────────────
    if "TotalCharges" in df.columns and "tenure" in df.columns:
        tenure_safe = df["tenure"].clip(lower=1)
        df["AvgMonthlyCharge"] = df["TotalCharges"] / tenure_safe

    if "MonthlyCharges" in df.columns and "AvgMonthlyCharge" in df.columns:
        df["ChargeIncrease"] = df["MonthlyCharges"] - df["AvgMonthlyCharge"]

    if "MonthlyCharges" in df.columns and "TotalCharges" in df.columns:
        total_safe = df["TotalCharges"].clip(lower=1)
        df["MonthlyToTotalRatio"] = df["MonthlyCharges"] / total_safe

    # ── Tenure-based features ────────────────────────────────────────────
    if "tenure" in df.columns:
        bins = [0, 12, 24, 48, 72, np.inf]
        labels = ["0-12", "13-24", "25-48", "49-72", "73+"]
        df["tenure_bin"] = pd.cut(
            df["tenure"], bins=bins, labels=labels, right=True
        )
        df["IsNewCustomer"] = (df["tenure"] <= 6).astype(int)

    # ── Service-count feature ────────────────────────────────────────────
    service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    present = [c for c in service_cols if c in df.columns]
    if present:
        df["NumServices"] = (
            df[present]
            .apply(lambda col: (col == "Yes").astype(int))
            .sum(axis=1)
        )

    # ── Binary convenience features ──────────────────────────────────────
    if "OnlineSecurity" in df.columns:
        df["HasSecurity"] = (df["OnlineSecurity"] == "Yes").astype(int)

    if "TechSupport" in df.columns:
        df["HasSupport"] = (df["TechSupport"] == "Yes").astype(int)

    if "PaymentMethod" in df.columns:
        df["AutoPay"] = (
            df["PaymentMethod"]
            .str.contains("automatic", case=False, na=False)
            .astype(int)
        )

    return df
