"""Pairwise diversity metrics and greedy diverse-subset selection.

Measures how differently base classifiers behave, which is the
theoretical foundation for why ensembles work — diverse models make
uncorrelated errors that cancel out during aggregation.

Metrics implemented:
    - Q-statistic      : Q ≈ 0 → high diversity, Q ≈ 1 → low diversity
    - Disagreement      : fraction of instances where two classifiers disagree
    - Double-fault      : fraction where both classifiers are wrong (lower = better)
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


# ── Prediction matrix ────────────────────────────────────────────────────────

def compute_predictions_matrix(
    pipelines: Dict[str, Any],
    X: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Binary predictions at *threshold* for every pipeline with predict_proba."""
    preds: Dict[str, np.ndarray] = {}
    for name, pipe in pipelines.items():
        if not hasattr(pipe, "predict_proba"):
            continue
        scores = pipe.predict_proba(X)[:, 1]
        preds[name] = (scores >= threshold).astype(int)
    return preds


# ── Pairwise diversity metrics ───────────────────────────────────────────────

def _contingency(
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    y_true: np.ndarray,
) -> Tuple[int, int, int, int]:
    """Return (N11, N10, N01, N00) contingency counts.

    N11 — both correct, N00 — both wrong,
    N10 — A correct / B wrong, N01 — A wrong / B correct.
    """
    correct_a = (y_pred_a == y_true).astype(int)
    correct_b = (y_pred_b == y_true).astype(int)

    n11 = int((correct_a & correct_b).sum())
    n10 = int((correct_a & ~correct_b.astype(bool)).sum())
    n01 = int((~correct_a.astype(bool) & correct_b).sum())
    n00 = int((~correct_a.astype(bool) & ~correct_b.astype(bool)).sum())
    return n11, n10, n01, n00


def q_statistic(
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Yule's Q-statistic.  Q → 0 means high diversity."""
    n11, n10, n01, n00 = _contingency(y_pred_a, y_pred_b, y_true)
    num = n11 * n00 - n01 * n10
    den = n11 * n00 + n01 * n10
    return num / den if den != 0 else 0.0


def disagreement_measure(
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> float:
    """Fraction of instances where the two classifiers disagree."""
    return float(np.mean(y_pred_a != y_pred_b))


def double_fault_measure(
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Fraction of instances where both classifiers are wrong."""
    both_wrong = (~(y_pred_a == y_true).astype(bool)) & (~(y_pred_b == y_true).astype(bool))
    return float(np.mean(both_wrong))


# ── Pairwise table ───────────────────────────────────────────────────────────

def compute_pairwise_diversity(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
) -> pd.DataFrame:
    """Compute all three diversity metrics for every unique model pair."""
    rows = []
    names = sorted(predictions.keys())
    for a, b in combinations(names, 2):
        rows.append({
            "model_a": a,
            "model_b": b,
            "q_statistic": q_statistic(predictions[a], predictions[b], y_true),
            "disagreement": disagreement_measure(predictions[a], predictions[b]),
            "double_fault": double_fault_measure(predictions[a], predictions[b], y_true),
        })
    return pd.DataFrame(rows)


# ── Greedy diverse-subset selection ──────────────────────────────────────────

def _avg_ensemble_f1(
    selected: List[str],
    pipelines: Dict[str, Any],
    X: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Soft-vote F1 for the selected subset (average probabilities)."""
    probas = np.column_stack([
        pipelines[n].predict_proba(X)[:, 1] for n in selected
    ])
    avg = probas.mean(axis=1)
    preds = (avg >= 0.5).astype(int)
    return float(f1_score(y_true, preds, zero_division=0))


def select_diverse_subset(
    pairwise_df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    pipelines: Dict[str, Any],
    X: np.ndarray,
    max_k: Optional[int] = None,
) -> List[str]:
    """Greedy forward selection that minimises avg Q-statistic.

    Starts with the single model having the highest individual F1.
    At each step, adds the model minimising average pairwise Q with the
    current set — but only if the soft-vote F1 on *y_true* improves.
    """
    names = sorted(predictions.keys())

    individual_f1 = {
        n: float(f1_score(y_true, predictions[n], zero_division=0))
        for n in names
    }
    best_start = max(individual_f1, key=individual_f1.get)
    selected = [best_start]
    remaining = [n for n in names if n != best_start]
    best_f1 = _avg_ensemble_f1(selected, pipelines, X, y_true)

    while remaining:
        if max_k is not None and len(selected) >= max_k:
            break

        # Build a quick lookup: avg Q-stat with current selected set
        candidates = []
        for cand in remaining:
            q_values = []
            for sel in selected:
                pair = pairwise_df[
                    ((pairwise_df["model_a"] == min(cand, sel))
                     & (pairwise_df["model_b"] == max(cand, sel)))
                ]
                if len(pair) > 0:
                    q_values.append(pair["q_statistic"].iloc[0])
            avg_q = np.mean(q_values) if q_values else 1.0
            candidates.append((cand, avg_q))

        # Pick the candidate with the lowest average Q (most diverse)
        candidates.sort(key=lambda x: x[1])
        added = False
        for cand, _ in candidates:
            trial = selected + [cand]
            trial_f1 = _avg_ensemble_f1(trial, pipelines, X, y_true)
            if trial_f1 > best_f1:
                selected.append(cand)
                remaining.remove(cand)
                best_f1 = trial_f1
                added = True
                break

        if not added:
            break

    return selected


# ── Evaluate the diverse subset on held-out data ─────────────────────────────

def evaluate_diverse_subset(
    selected_names: List[str],
    pipelines: Dict[str, Any],
    X: np.ndarray,
    y_true: np.ndarray,
) -> Dict[str, Any]:
    """Average-probability ensemble from the selected models."""
    from Common.eval import evaluate_binary

    probas = np.column_stack([
        pipelines[n].predict_proba(X)[:, 1] for n in selected_names
    ])
    avg_scores = probas.mean(axis=1)
    return evaluate_binary(y_true, avg_scores, threshold=0.5)


# ── Visualisation ────────────────────────────────────────────────────────────

def plot_diversity_heatmap(
    pairwise_df: pd.DataFrame,
    metric: str,
    save_path: str | Path,
) -> None:
    """Square heatmap of a pairwise diversity metric."""
    pivot = pairwise_df.pivot(index="model_a", columns="model_b", values=metric)

    # Make symmetric
    names = sorted(set(pairwise_df["model_a"]) | set(pairwise_df["model_b"]))
    matrix = pd.DataFrame(np.nan, index=names, columns=names)
    for _, row in pairwise_df.iterrows():
        matrix.loc[row["model_a"], row["model_b"]] = row[metric]
        matrix.loc[row["model_b"], row["model_a"]] = row[metric]
    vals = matrix.values.copy()
    np.fill_diagonal(vals, 0.0)
    matrix = pd.DataFrame(vals, index=names, columns=names)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix.values.astype(float), cmap="RdYlGn_r", aspect="auto")

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)

    # Annotate cells
    for i in range(len(names)):
        for j in range(len(names)):
            val = matrix.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7)

    pretty = metric.replace("_", " ").title()
    ax.set_title(f"Pairwise {pretty}", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[REPORT] Diversity heatmap ({metric}) → {save_path}")
