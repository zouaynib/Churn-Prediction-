from __future__ import annotations

import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from Common.eval import predict_at_threshold
from Common.preprocess import preprocess_data
from Common.config import (
    TARGET_COL,
    DEFAULT_THRESHOLD,
    SCALE_NUMERIC,
    SCALER_TYPE,
    STRATIFY,
    TEST_SIZE,
    VAL_SIZE,
)


def load_pipeline(model_path: str | Path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def save_predictions(df_out: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Churn inference")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved pipeline .pkl/.joblib")
    parser.add_argument("--data-path", type=str, required=True, help="Path to input CSV (new customers)")
    parser.add_argument("--out-path", type=str, default="outputs/predictions.csv", help="Where to save predictions CSV")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Decision threshold for churn")
    parser.add_argument("--id-col", type=str, default=None, help="Optional ID column to keep in output")

    args = parser.parse_args()

    pipeline = load_pipeline(args.model_path)

    # Load raw data
    df = pd.read_csv(args.data_path)

    # If TARGET_COL exists in this file, drop it for inference
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    # Reuse your preprocessing function in "inference mode"
    # We call preprocess_data but we only want the transformed X (no splitting).
    # So we mimic your pipeline expectations: we pass a temp CSV and ask it to return X_train as "all data".
    #
    # >>> If your preprocess_data already has a function like preprocess_inference(df),
    #     use it instead (preferred).
    #
    # Here is a safe fallback: we will call preprocess_data by writing a temp file.
    tmp_path = Path("outputs/_tmp_infer.csv")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp_path, index=False)

    data = preprocess_data(
        str(tmp_path),
        target_col=TARGET_COL,          # absent now, so it should just treat it as features
        test_size=TEST_SIZE,            # ignored in inference use if your function checks target existence
        val_size=VAL_SIZE,
        stratify=STRATIFY,
        scale_numeric=SCALE_NUMERIC,
        scaler_type=SCALER_TYPE,
    )

    # Depending on your preprocess_data design, one of these will be "all features"
    # Most common: data.X_train contains features after preprocessing.
    X = data.X_train

    # Predict probabilities
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)[:, 1]
    else:
        # fallback: decision_function / predict
        proba = pipeline.predict(X).astype(float)

    y_pred = predict_at_threshold(proba, args.threshold)

    # Build output
    out = pd.DataFrame({
        "churn_proba": proba,
        "churn_pred": y_pred,
    })

    if args.id_col and args.id_col in df.columns:
        out.insert(0, args.id_col, df[args.id_col].values)

    save_predictions(out, args.out_path)

    # Print a small summary
    print(json.dumps({
        "model_path": args.model_path,
        "data_path": args.data_path,
        "out_path": args.out_path,
        "threshold": args.threshold,
        "n_rows": int(len(out)),
        "predicted_churn_rate": float(out["churn_pred"].mean()),
    }, indent=2))


if __name__ == "__main__":
    main()
