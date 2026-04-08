from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def compute_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fn: float = 500.0,
    cost_fp: float = 50.0,
) -> float:
    """Total business cost for a set of predictions.

    Parameters
    ----------
    y_true   : ground-truth labels (0/1)
    y_pred   : predicted labels   (0/1)
    cost_fn  : cost of a false negative (missed churner — lost lifetime value)
    cost_fp  : cost of a false positive (wasted retention offer)
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())

    return fn * cost_fn + fp * cost_fp


def find_business_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    cost_fn: float = 500.0,
    cost_fp: float = 50.0,
    grid: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Return (threshold, cost) that minimises total business cost."""
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)

    best_t, best_cost = 0.5, float("inf")
    for t in grid:
        y_pred = (scores >= t).astype(int)
        cost = compute_cost(y_true, y_pred, cost_fn, cost_fp)
        if cost < best_cost:
            best_cost = cost
            best_t = float(t)

    return best_t, best_cost


def cost_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    cost_fn: float = 500.0,
    cost_fp: float = 50.0,
    grid: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Return per-threshold costs and breakdown (for plotting)."""
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)

    costs, fn_counts, fp_counts = [], [], []

    for t in grid:
        y_pred = (scores >= t).astype(int)
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn_counts.append(fn)
        fp_counts.append(fp)
        costs.append(fn * cost_fn + fp * cost_fp)

    return {
        "thresholds": np.array(grid),
        "costs": np.array(costs),
        "fn_counts": np.array(fn_counts),
        "fp_counts": np.array(fp_counts),
    }
