"""Calibration analysis — reliability diagrams, ECE, and calibrated stacking.

Well-calibrated probabilities are critical for:
    1. Cost-sensitive threshold optimization (the whole business-cost analysis)
    2. Stacking — the meta-learner receives base-learner probabilities, so
       miscalibrated inputs degrade the meta-learner's decision surface.

This module answers: *which models are well-calibrated, does post-hoc
calibration help, and does feeding calibrated probabilities to stacking
improve the final ensemble?*
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss


# ── Expected Calibration Error ───────────────────────────────────────────────

def compute_ece(
    y_true: np.ndarray,
    probas: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (weighted bin-level |acc − conf|)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probas > lo) & (probas <= hi)
        count = mask.sum()
        if count == 0:
            continue
        avg_conf = probas[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += (count / n) * abs(avg_acc - avg_conf)

    return float(ece)


# ── Calibrate a single model ────────────────────────────────────────────────

def calibrate_model(
    pipeline: Any,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    """Wrap a fitted pipeline in CalibratedClassifierCV(cv='prefit')."""
    cal = CalibratedClassifierCV(pipeline, cv="prefit", method=method)
    cal.fit(X_cal, y_cal)
    return cal


# ── Calibrate all models ────────────────────────────────────────────────────

def calibrate_all_models(
    pipelines: Dict[str, Any],
    X_cal: np.ndarray,
    y_cal: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """For every model with predict_proba, produce Platt + isotonic wrappers.

    Returns {name: {"original": pipe, "platt": cal, "isotonic": cal}}.
    """
    calibrated: Dict[str, Dict[str, Any]] = {}

    for name, pipe in pipelines.items():
        if not hasattr(pipe, "predict_proba"):
            continue
        calibrated[name] = {
            "original": pipe,
            "platt": calibrate_model(pipe, X_cal, y_cal, method="sigmoid"),
            "isotonic": calibrate_model(pipe, X_cal, y_cal, method="isotonic"),
        }

    return calibrated


# ── Summary table (ECE + Brier) ─────────────────────────────────────────────

def compute_calibration_summary(
    calibrated_models: Dict[str, Dict[str, Any]],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """ECE and Brier score for each model × variant (original / platt / isotonic)."""
    rows = []
    for name, variants in calibrated_models.items():
        for variant_name, model in variants.items():
            probas = model.predict_proba(X_test)[:, 1]
            rows.append({
                "model": name,
                "variant": variant_name,
                "ece": compute_ece(y_test, probas),
                "brier_score": brier_score_loss(y_test, probas),
            })
    return pd.DataFrame(rows)


# ── Reliability diagrams ────────────────────────────────────────────────────

def plot_reliability_diagrams(
    calibrated_models: Dict[str, Dict[str, Any]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_dir: str | Path,
    n_bins: int = 10,
) -> None:
    """One figure per model with original / Platt / isotonic curves."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_names = sorted(calibrated_models.keys())

    # Per-model figures
    for name in model_names:
        variants = calibrated_models[name]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfectly calibrated")

        for v_name, style in [
            ("original", "-"),
            ("platt", "--"),
            ("isotonic", "-."),
        ]:
            model = variants[v_name]
            probas = model.predict_proba(X_test)[:, 1]
            frac_pos, mean_pred = calibration_curve(y_test, probas, n_bins=n_bins)
            ece = compute_ece(y_test, probas)
            ax.plot(
                mean_pred, frac_pos,
                linestyle=style, marker="o", markersize=4,
                label=f"{v_name} (ECE={ece:.4f})",
            )

        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"Reliability Diagram — {name}")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / f"calibration_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Overview grid
    n = len(model_names)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for idx, name in enumerate(model_names):
        ax = axes_flat[idx]
        variants = calibrated_models[name]
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)

        for v_name, style in [("original", "-"), ("platt", "--"), ("isotonic", "-.")]:
            probas = variants[v_name].predict_proba(X_test)[:, 1]
            frac_pos, mean_pred = calibration_curve(y_test, probas, n_bins=n_bins)
            ece = compute_ece(y_test, probas)
            ax.plot(mean_pred, frac_pos, linestyle=style, marker="o", markersize=3,
                    label=f"{v_name} ({ece:.3f})")

        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Calibration Overview (ECE in legend)", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_dir / "calibration_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[REPORT] Reliability diagrams → {save_dir}/calibration_*.png")


# ── Calibrated stacking comparison ──────────────────────────────────────────

def build_calibrated_stacking(
    calibrated_models: Dict[str, Dict[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    base_names: Tuple[str, ...] = (
        "lightgbm_model", "xgboost", "gradientboosting", "randomforest",
    ),
    method: str = "isotonic",
) -> Dict[str, Any]:
    """Re-build stacking using calibrated base learners and compare.

    Uses CalibratedClassifierCV(cv=5) on *fresh* base learners so that
    stacking's internal CV can fit+calibrate each fold properly.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression

    base_learners = []
    for bname in base_names:
        if bname not in calibrated_models:
            continue
        original_pipe = calibrated_models[bname]["original"]

        # Unwrap Pipeline([("model", clf)]) to get the raw estimator
        if hasattr(original_pipe, "named_steps") and "model" in original_pipe.named_steps:
            base_clf = original_pipe.named_steps["model"]
        else:
            base_clf = original_pipe

        # Wrap in calibrator with cv=5 (NOT prefit)
        cal_clf = CalibratedClassifierCV(base_clf, cv=5, method=method)
        base_learners.append((bname, cal_clf))

    if not base_learners:
        print("[WARN] No base learners available for calibrated stacking")
        return {}

    meta = LogisticRegression(
        C=1.0, class_weight="balanced", random_state=42,
        max_iter=1000, solver="lbfgs",
    )

    stack = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta,
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )

    print("  Fitting calibrated stacking …")
    stack.fit(X_train, y_train)

    from Common.eval import evaluate_binary
    test_scores = stack.predict_proba(X_test)[:, 1]
    metrics = evaluate_binary(y_test, test_scores, threshold=0.5)

    return {"model": stack, "metrics": metrics, "method": method}


def print_stacking_comparison(
    original_metrics: Dict[str, Any],
    calibrated_metrics: Dict[str, Any],
) -> None:
    """Side-by-side comparison: original vs calibrated stacking."""
    keys = ["f1", "recall", "precision", "roc_auc", "pr_auc"]
    print("\n  Stacking: Original vs Calibrated Base Learners")
    print(f"  {'Metric':<12} {'Original':>10} {'Calibrated':>10} {'Delta':>8}")
    print(f"  {'-'*42}")
    for k in keys:
        orig = original_metrics.get(k, 0)
        cal = calibrated_metrics.get(k, 0)
        delta = cal - orig
        sign = "+" if delta >= 0 else ""
        print(f"  {k:<12} {orig:>10.4f} {cal:>10.4f} {sign}{delta:>7.4f}")
