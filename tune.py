"""tune.py — Optuna hyperparameter tuning for top models.

Tunes LightGBM, XGBoost, and RandomForest using Stratified 5-fold CV
optimising F1. Prints best params at the end.

Usage:
    pip install optuna          # if not already installed
    python tune.py              # runs 50 trials per model (~5-10 min)
    python tune.py --trials 100 # more trials for better results
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score

from Common.config import (
    DATA_PATH,
    SCALE_NUMERIC,
    SCALER_TYPE,
    STRATIFY,
    TARGET_COL,
    TEST_SIZE,
    VAL_SIZE,
)
from Common.preprocess import preprocess_data

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

ARTIFACTS_DIR = Path("artifacts")


# ── Objective factories ──────────────────────────────────────────────────────

def lgbm_objective(X, y):
    import lightgbm as lgb

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "class_weight": "balanced",
            "random_state": 42,
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1)
        return scores.mean()

    return objective


def xgb_objective(X, y):
    from xgboost import XGBClassifier

    neg = (y == 0).sum()
    pos = (y == 1).sum()
    ratio = neg / max(pos, 1)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, ratio * 1.5),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        }
        model = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1)
        return scores.mean()

    return objective


def rf_objective(X, y):
    from sklearn.ensemble import RandomForestClassifier

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 4, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
        model = RandomForestClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1)
        return scores.mean()

    return objective


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tune top models with Optuna")
    parser.add_argument("--trials", type=int, default=50, help="Trials per model")
    args = parser.parse_args()

    print("Loading and preprocessing data...")
    data = preprocess_data(
        DATA_PATH,
        target_col=TARGET_COL,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        stratify=STRATIFY,
        scale_numeric=SCALE_NUMERIC,
        scaler_type=SCALER_TYPE,
    )

    # Combine train + val for CV tuning
    X = np.vstack([data.X_train, data.X_val])
    y = np.concatenate([data.y_train, data.y_val])

    results = {}

    tuners = [
        ("LightGBM", lgbm_objective),
        ("XGBoost", xgb_objective),
        ("RandomForest", rf_objective),
    ]

    for name, objective_fn in tuners:
        print(f"\n{'=' * 60}")
        print(f"  Tuning {name}  ({args.trials} trials)")
        print(f"{'=' * 60}")

        study = optuna.create_study(direction="maximize")
        study.optimize(objective_fn(X, y), n_trials=args.trials, show_progress_bar=True)

        print(f"  Best F1 (CV): {study.best_value:.4f}")
        print(f"  Best params:")
        for k, v in study.best_params.items():
            print(f"    {k}: {v}")

        results[name] = {
            "best_f1_cv": study.best_value,
            "best_params": study.best_params,
        }

    # Save results
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    out_path = ARTIFACTS_DIR / "tuning_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[SAVED] {out_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("  TUNING SUMMARY")
    print(f"{'=' * 60}")
    for name, r in results.items():
        print(f"  {name:15s}  CV F1 = {r['best_f1_cv']:.4f}")
    print(f"\nCopy the best_params into models/*.py and re-run train.py")


if __name__ == "__main__":
    main()
