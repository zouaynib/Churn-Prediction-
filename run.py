from __future__ import annotations

import importlib
from typing import Dict, Any

import numpy as np
from sklearn.pipeline import Pipeline

from Common.config import (
    DATA_PATH,
    DEFAULT_THRESHOLD,
    SCALE_NUMERIC,
    SCALER_TYPE,
    STRATIFY,
    TARGET_COL,
    TEST_SIZE,
    VAL_SIZE,
)
from Common.eval import evaluate_binary, tune_threshold
from Common.preprocess import preprocess_data


MODEL_MODULES = [
    "models.adaboost",
    "models.bagging",
    "models.extra_trees",
    "models.gradientboosting",
    "models.lightgbm_model",
    "models.randomforest",
    "models.stacking",
    "models.voting_hard",
    "models.voting_soft",
    "models.xgboost",
]


def load_builder(module_path: str):
    mod = importlib.import_module(module_path)
    if not hasattr(mod, "build_model"):
        raise AttributeError(f"{module_path} has no function build_model()")
    return getattr(mod, "build_model")


def main() -> None:
    data = preprocess_data(
        DATA_PATH,
        target_col=TARGET_COL,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        stratify=STRATIFY,
        scale_numeric=SCALE_NUMERIC,
        scaler_type=SCALER_TYPE,
    )

    grid = np.linspace(0.001, 0.999, 999)

    results: Dict[str, Any] = {}

    for module_path in MODEL_MODULES:
        name = module_path.split(".")[-1]

        print(f"\n{'=' * 70}")
        print(f"Running: {name}")
        print(f"{'=' * 70}")

        build_fn = load_builder(module_path)
        model = build_fn()

        pipeline = Pipeline([("model", model)])
        pipeline.fit(data.X_train, data.y_train)

        val_scores = pipeline.predict_proba(data.X_val)[:, 1]
        t_f1, v_f1 = tune_threshold(data.y_val, val_scores, metric="f1", grid=grid)
        t_mcc, v_mcc = tune_threshold(data.y_val, val_scores, metric="mcc", grid=grid)

        print(f"[VAL] best F1  threshold = {t_f1:.3f} | F1  = {v_f1:.4f}")
        print(f"[VAL] best MCC threshold = {t_mcc:.3f} | MCC = {v_mcc:.4f}")

        test_scores = pipeline.predict_proba(data.X_test)[:, 1]

        rep_default = evaluate_binary(data.y_test, test_scores, threshold=DEFAULT_THRESHOLD)
        rep_f1 = evaluate_binary(data.y_test, test_scores, threshold=t_f1)
        rep_mcc = evaluate_binary(data.y_test, test_scores, threshold=t_mcc)

        results[name] = {
            "val": {"t_f1": t_f1, "f1": v_f1, "t_mcc": t_mcc, "mcc": v_mcc},
            "test": {"default": rep_default, "f1": rep_f1, "mcc": rep_mcc},
        }

        print(
            f"[TEST] default F1={rep_default['f1']:.3f} | "
            f"F1@t_f1={rep_f1['f1']:.3f} | "
            f"F1@t_mcc={rep_mcc['f1']:.3f} | "
            f"ROC_AUC={rep_default['roc_auc']:.3f}"
        )

    print("\n\n===== OVERALL SUMMARY (sorted by test F1@t_f1) =====")
    summary = []
    for name, r in results.items():
        summary.append(
            (name, r["test"]["f1"]["f1"], r["test"]["mcc"]["f1"], r["test"]["default"]["roc_auc"])
        )
    summary.sort(key=lambda x: x[1], reverse=True)

    for name, f1_at_f1, f1_at_mcc, roc_auc in summary:
        print(f"{name:16s} | F1@t_f1={f1_at_f1:.3f} | F1@t_mcc={f1_at_mcc:.3f} | ROC_AUC={roc_auc:.3f}")


if __name__ == "__main__":
    main()
