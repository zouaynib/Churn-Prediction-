from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve

from Common.cost_analysis import cost_curve


# ── ROC curves ───────────────────────────────────────────────────────────────

def plot_roc_curves(
    results: Dict[str, Any],
    y_test: np.ndarray,
    test_scores: Dict[str, np.ndarray],
    save_path: str | Path = "reports/roc_curves.png",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    for name, scores in test_scores.items():
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = results[name]["test"]["default"]["roc_auc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", linewidth=1.5)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[REPORT] ROC curves → {save_path}")


# ── Precision–Recall curves ──────────────────────────────────────────────────

def plot_pr_curves(
    results: Dict[str, Any],
    y_test: np.ndarray,
    test_scores: Dict[str, np.ndarray],
    save_path: str | Path = "reports/pr_curves.png",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    for name, scores in test_scores.items():
        prec, rec, _ = precision_recall_curve(y_test, scores)
        pr_auc = results[name]["test"]["default"]["pr_auc"]
        ax.plot(rec, prec, label=f"{name} (AP={pr_auc:.3f})", linewidth=1.5)

    baseline = float(y_test.mean())
    ax.axhline(
        baseline, color="k", linestyle="--", linewidth=0.8,
        label=f"Baseline ({baseline:.2f})"
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves — All Models")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[REPORT] PR curves → {save_path}")


# ── Business cost curve ──────────────────────────────────────────────────────

def plot_cost_analysis(
    y_test: np.ndarray,
    scores: np.ndarray,
    model_name: str,
    cost_fn: float = 500.0,
    cost_fp: float = 50.0,
    save_path: str | Path = "reports/cost_analysis.png",
) -> None:
    data = cost_curve(y_test, scores, cost_fn=cost_fn, cost_fp=cost_fp)
    best_idx = int(np.argmin(data["costs"]))
    best_t = float(data["thresholds"][best_idx])
    best_cost = float(data["costs"][best_idx])

    # Default-threshold cost for comparison
    default_idx = int(np.argmin(np.abs(data["thresholds"] - 0.5)))
    default_cost = float(data["costs"][default_idx])
    savings = default_cost - best_cost

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(data["thresholds"], data["costs"], color="steelblue", linewidth=2, label="Total cost")
    ax.axvline(
        best_t, color="red", linestyle="--",
        label=f"Optimal t={best_t:.2f}  (${best_cost:,.0f})"
    )
    ax.axvline(
        0.5, color="gray", linestyle=":",
        label=f"Default t=0.5  (${default_cost:,.0f})"
    )
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Total Business Cost ($)")
    ax.set_title(
        f"Business Cost vs Threshold — {model_name}\n"
        f"FN cost=${cost_fn:,.0f}  |  FP cost=${cost_fp:,.0f}  |  "
        f"Savings vs default: ${savings:,.0f}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[REPORT] Cost analysis → {save_path}")


# ── Rich summary table ───────────────────────────────────────────────────────

def print_rich_summary(results: Dict[str, Any]) -> None:
    """Print a coloured comparison table (falls back to plain text if rich absent)."""

    rows = []
    for name, r in results.items():
        d = r["test"]
        f1_val = d["f1"].get("f1")
        recall_val = d["f1"].get("recall")
        prec_val = d["f1"].get("precision")
        biz_t = r.get("business", {}).get("threshold")
        biz_cost = r.get("business", {}).get("cost")
        rows.append((name, d["default"]["roc_auc"], d["default"]["pr_auc"],
                     f1_val, recall_val, prec_val, biz_t, biz_cost))

    rows.sort(key=lambda x: (x[3] or 0), reverse=True)

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        tbl = Table(title="Model Comparison", header_style="bold cyan", show_lines=True)
        tbl.add_column("Model", style="bold")
        tbl.add_column("ROC AUC", justify="right")
        tbl.add_column("PR AUC", justify="right")
        tbl.add_column("F1@t_f1", justify="right")
        tbl.add_column("Recall", justify="right")
        tbl.add_column("Precision", justify="right")
        tbl.add_column("Biz threshold", justify="right")
        tbl.add_column("Biz cost ($)", justify="right")

        for name, roc, pr, f1, rec, prec, bt, bc in rows:
            tbl.add_row(
                name,
                f"{roc:.4f}",
                f"{pr:.4f}",
                f"{f1:.4f}" if f1 is not None else "N/A",
                f"{rec:.4f}" if rec is not None else "N/A",
                f"{prec:.4f}" if prec is not None else "N/A",
                f"{bt:.2f}" if bt is not None else "N/A",
                f"{bc:,.0f}" if bc is not None else "N/A",
            )
        console.print(tbl)

    except ImportError:
        header = (
            f"{'Model':20s} | {'ROC AUC':>8} | {'PR AUC':>8} | "
            f"{'F1@t_f1':>8} | {'Recall':>8} | {'Precision':>8} | "
            f"{'Biz t':>6} | {'Biz cost':>10}"
        )
        print("\n" + header)
        print("-" * len(header))
        for name, roc, pr, f1, rec, prec, bt, bc in rows:
            f1_s = f"{f1:.4f}" if f1 is not None else "N/A"
            rec_s = f"{rec:.4f}" if rec is not None else "N/A"
            prec_s = f"{prec:.4f}" if prec is not None else "N/A"
            bt_s = f"{bt:.2f}" if bt is not None else "N/A"
            bc_s = f"{bc:,.0f}" if bc is not None else "N/A"
            print(
                f"{name:20s} | {roc:>8.4f} | {pr:>8.4f} | "
                f"{f1_s:>8} | {rec_s:>8} | {prec_s:>8} | "
                f"{bt_s:>6} | {bc_s:>10}"
            )
