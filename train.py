"""train.py — full training pipeline.

Runs all models, tunes thresholds (F1, MCC, business cost),
saves the best model + preprocessor as artifacts, and generates
ROC / PR / cost-analysis reports.

Usage:
    python train.py

Artifacts saved to  artifacts/
Reports saved to    reports/
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Dict

import joblib
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
from Common.cost_analysis import find_business_threshold
from Common.eval import evaluate_binary, tune_threshold
from Common.preprocess import preprocess_data
from Common.reporting import (
    plot_cost_analysis,
    plot_pr_curves,
    plot_roc_curves,
    print_rich_summary,
)
from Common.diversity import (
    compute_predictions_matrix,
    compute_pairwise_diversity,
    evaluate_diverse_subset,
    plot_diversity_heatmap,
    select_diverse_subset,
)
from Common.calibration import (
    build_calibrated_stacking,
    calibrate_all_models,
    compute_calibration_summary,
    plot_reliability_diagrams,
    print_stacking_comparison,
)

# ── Business cost parameters ─────────────────────────────────────────────────
# Adjust these to match your real-world estimates.
COST_FN = 500.0   # $ lost when a churner is missed (lost lifetime value)
COST_FP = 50.0    # $ wasted on a retention offer for a loyal customer

ARTIFACTS_DIR = Path("artifacts")
REPORTS_DIR = Path("reports")

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
        raise AttributeError(f"{module_path} has no build_model()")
    return getattr(mod, "build_model")


def main() -> None:
    # ── Data ─────────────────────────────────────────────────────────────────
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
    test_scores_all: Dict[str, np.ndarray] = {}   # name → proba array (for plots)
    pipelines: Dict[str, Pipeline] = {}

    # ── Train every model ────────────────────────────────────────────────────
    for module_path in MODEL_MODULES:
        name = module_path.split(".")[-1]

        print(f"\n{'=' * 70}")
        print(f"  {name}")
        print(f"{'=' * 70}")

        build_fn = load_builder(module_path)
        model = build_fn()

        pipeline = Pipeline([("model", model)])
        pipeline.fit(data.X_train, data.y_train)
        pipelines[name] = pipeline

        # Hard-voting has no predict_proba → evaluate at 0.5 only
        if not hasattr(pipeline, "predict_proba"):
            print("[INFO] No predict_proba — skipping threshold tuning")
            test_preds = pipeline.predict(data.X_test)
            rep = evaluate_binary(data.y_test, test_preds.astype(float), threshold=DEFAULT_THRESHOLD)
            results[name] = {
                "val": {"t_f1": None, "f1": None, "t_mcc": None, "mcc": None},
                "test": {"default": rep, "f1": rep, "mcc": rep},
            }
            print(f"[TEST] F1={rep['f1']:.3f}  (hard-voting, no proba)")
            continue

        # Threshold tuning on validation set
        val_scores = pipeline.predict_proba(data.X_val)[:, 1]
        t_f1, v_f1 = tune_threshold(data.y_val, val_scores, metric="f1", grid=grid)
        t_mcc, v_mcc = tune_threshold(data.y_val, val_scores, metric="mcc", grid=grid)

        print(f"[VAL] best F1  threshold = {t_f1:.3f} | F1  = {v_f1:.4f}")
        print(f"[VAL] best MCC threshold = {t_mcc:.3f} | MCC = {v_mcc:.4f}")

        # Test-set scores
        test_scores = pipeline.predict_proba(data.X_test)[:, 1]
        test_scores_all[name] = test_scores

        # Business-optimal threshold
        t_biz, biz_cost = find_business_threshold(
            data.y_test, test_scores, cost_fn=COST_FN, cost_fp=COST_FP
        )
        print(f"[BIZ] optimal threshold = {t_biz:.3f} | total cost = ${biz_cost:,.0f}")

        rep_default = evaluate_binary(data.y_test, test_scores, threshold=DEFAULT_THRESHOLD)
        rep_f1 = evaluate_binary(data.y_test, test_scores, threshold=t_f1)
        rep_mcc = evaluate_binary(data.y_test, test_scores, threshold=t_mcc)
        rep_biz = evaluate_binary(data.y_test, test_scores, threshold=t_biz)

        results[name] = {
            "val": {"t_f1": t_f1, "f1": v_f1, "t_mcc": t_mcc, "mcc": v_mcc},
            "test": {
                "default": rep_default,
                "f1": rep_f1,
                "mcc": rep_mcc,
                "biz": rep_biz,
            },
            "business": {"threshold": t_biz, "cost": biz_cost},
        }

        print(
            f"[TEST] default F1={rep_default['f1']:.3f} | "
            f"F1@t_f1={rep_f1['f1']:.3f} | "
            f"F1@t_mcc={rep_mcc['f1']:.3f} | "
            f"ROC_AUC={rep_default['roc_auc']:.3f}"
        )

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n")
    print_rich_summary(results)

    # ── Pick best model by F1@t_f1 ───────────────────────────────────────────
    scored = [
        (name, r["test"]["f1"].get("f1", 0) or 0)
        for name, r in results.items()
    ]
    best_name, best_f1 = max(scored, key=lambda x: x[1])
    print(f"\n[BEST MODEL] {best_name}  F1@t_f1={best_f1:.4f}")

    # ── Save artifacts ────────────────────────────────────────────────────────
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    joblib.dump(pipelines[best_name], ARTIFACTS_DIR / "best_model.joblib")
    joblib.dump(data.preprocessor, ARTIFACTS_DIR / "preprocessor.joblib")

    (ARTIFACTS_DIR / "feature_names.json").write_text(
        json.dumps(data.feature_names, indent=2)
    )
    (ARTIFACTS_DIR / "results.json").write_text(
        json.dumps(results, indent=2, default=str)
    )

    best_info = {
        "name": best_name,
        "threshold_f1": results[best_name]["val"]["t_f1"],
        "threshold_mcc": results[best_name]["val"]["t_mcc"],
        "threshold_biz": results[best_name].get("business", {}).get("threshold", 0.5),
        "cost_fn": COST_FN,
        "cost_fp": COST_FP,
    }
    (ARTIFACTS_DIR / "best_model_info.json").write_text(
        json.dumps(best_info, indent=2)
    )
    print(f"[ARTIFACTS] saved to {ARTIFACTS_DIR}/")

    # ── Generate reports ──────────────────────────────────────────────────────
    REPORTS_DIR.mkdir(exist_ok=True)

    if test_scores_all:
        plot_roc_curves(results, data.y_test, test_scores_all, REPORTS_DIR / "roc_curves.png")
        plot_pr_curves(results, data.y_test, test_scores_all, REPORTS_DIR / "pr_curves.png")

    if best_name in test_scores_all:
        plot_cost_analysis(
            data.y_test,
            test_scores_all[best_name],
            model_name=best_name,
            cost_fn=COST_FN,
            cost_fp=COST_FP,
            save_path=REPORTS_DIR / "cost_analysis.png",
        )

    print(f"[REPORTS]   saved to {REPORTS_DIR}/")

    # ── Diversity analysis ───────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  DIVERSITY ANALYSIS")
    print(f"{'=' * 70}")

    val_preds = compute_predictions_matrix(pipelines, data.X_val, threshold=0.5)
    pairwise_df = compute_pairwise_diversity(val_preds, data.y_val)
    pairwise_df.to_csv(REPORTS_DIR / "diversity_pairwise.csv", index=False)

    for metric in ("q_statistic", "disagreement", "double_fault"):
        plot_diversity_heatmap(
            pairwise_df, metric,
            save_path=REPORTS_DIR / f"diversity_{metric}.png",
        )

    diverse_subset = select_diverse_subset(
        pairwise_df, val_preds, data.y_val,
        pipelines=pipelines, X=data.X_val,
    )
    print(f"\n[DIVERSE SUBSET] {diverse_subset}")

    subset_metrics = evaluate_diverse_subset(
        diverse_subset, pipelines, data.X_test, data.y_test,
    )
    print(
        f"[DIVERSE SUBSET TEST] F1={subset_metrics['f1']:.4f}  "
        f"Recall={subset_metrics['recall']:.4f}  "
        f"ROC_AUC={subset_metrics['roc_auc']:.4f}"
    )
    print(f"[BEST SINGLE MODEL]  F1={best_f1:.4f}  ({best_name})")

    (ARTIFACTS_DIR / "diverse_subset.json").write_text(
        json.dumps({"models": diverse_subset, "test_metrics": subset_metrics}, indent=2, default=str)
    )

    # ── Dynamic Ensemble Selection (DES) ─────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  DYNAMIC ENSEMBLE SELECTION (DES)")
    print(f"{'=' * 70}")

    try:
        from Common.des import (
            build_pool_of_classifiers,
            print_des_summary,
            run_des_methods,
        )

        pool_names, pool_clfs = build_pool_of_classifiers(pipelines)
        print(f"  Pool: {pool_names}")

        des_results = run_des_methods(
            pool_clfs, data.X_val, data.y_val, data.X_test, data.y_test,
        )
        print_des_summary(des_results)

        (ARTIFACTS_DIR / "des_results.json").write_text(
            json.dumps(des_results, indent=2, default=str)
        )
    except ImportError as exc:
        print(f"[WARN] DESlib not installed — skipping DES ({exc})")
        print("       Install with: pip install deslib")

    # ── Calibration analysis ─────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  CALIBRATION ANALYSIS")
    print(f"{'=' * 70}")

    calibrated_models = calibrate_all_models(pipelines, data.X_val, data.y_val)

    cal_summary = compute_calibration_summary(calibrated_models, data.X_test, data.y_test)
    cal_summary.to_csv(REPORTS_DIR / "calibration_summary.csv", index=False)
    print("\n  ECE & Brier scores (lower = better):")
    print(cal_summary.to_string(index=False))

    plot_reliability_diagrams(calibrated_models, data.X_test, data.y_test, REPORTS_DIR)

    # Calibrated stacking comparison
    if "stacking" in results:
        cal_stack = build_calibrated_stacking(
            calibrated_models, data.X_train, data.y_train,
            data.X_test, data.y_test,
        )
        if cal_stack:
            from Common.eval import evaluate_binary as _eval
            original_stack_metrics = results["stacking"]["test"]["default"]
            print_stacking_comparison(original_stack_metrics, cal_stack["metrics"])

            (ARTIFACTS_DIR / "calibrated_stacking_results.json").write_text(
                json.dumps(cal_stack["metrics"], indent=2, default=str)
            )

    print(f"\n{'=' * 70}")
    print("  DONE — all analyses complete")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
