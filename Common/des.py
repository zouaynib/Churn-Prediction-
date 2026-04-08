"""Dynamic Ensemble Selection (DES) via DESlib.

Instead of using a fixed ensemble for all test instances, DES methods
select *different* classifiers per instance based on local competence
in the region of competence (k-nearest neighbours in the validation set).

Methods:
    - KNORA-E  (K-Nearest Oracles Eliminate)   — keeps only locally correct classifiers
    - KNORA-U  (K-Nearest Oracles Union)        — weights by local accuracy
    - META-DES (META-learning for DES)          — learns a meta-classifier to select
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from Common.eval import evaluate_binary


def build_pool_of_classifiers(
    pipelines: Dict[str, Any],
    exclude: Tuple[str, ...] = ("stacking", "voting_hard", "voting_soft"),
) -> Tuple[List[str], List[Any]]:
    """Extract base classifiers that have predict_proba.

    Returns (names, estimators) — estimators are unwrapped from the
    Pipeline([("model", clf)]) wrapper.
    """
    names: List[str] = []
    estimators: List[Any] = []

    for name, pipe in sorted(pipelines.items()):
        if name in exclude:
            continue
        if not hasattr(pipe, "predict_proba"):
            continue

        # Unwrap Pipeline([("model", clf)]) → clf
        if hasattr(pipe, "named_steps") and "model" in pipe.named_steps:
            clf = pipe.named_steps["model"]
        else:
            clf = pipe

        names.append(name)
        estimators.append(clf)

    return names, estimators


def run_des_methods(
    pool_classifiers: List[Any],
    X_dsel: np.ndarray,
    y_dsel: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k: int = 7,
) -> Dict[str, Dict[str, Any]]:
    """Fit and evaluate KNORA-E, KNORA-U, META-DES on the test set.

    Parameters
    ----------
    pool_classifiers : list of fitted classifiers
    X_dsel / y_dsel  : dynamic selection (validation) set
    X_test / y_test  : held-out test set
    k                : neighbourhood size for competence estimation
    """
    from deslib.des import KNORAE, KNORAU, METADES

    methods = {
        "KNORA-E": KNORAE(pool_classifiers=pool_classifiers, k=k, random_state=42),
        "KNORA-U": KNORAU(pool_classifiers=pool_classifiers, k=k, random_state=42),
        "META-DES": METADES(pool_classifiers=pool_classifiers, k=k, random_state=42),
    }

    results: Dict[str, Dict[str, Any]] = {}

    for method_name, des in methods.items():
        print(f"  Fitting {method_name} …")
        des.fit(X_dsel, y_dsel)

        test_scores = des.predict_proba(X_test)[:, 1]
        metrics = evaluate_binary(y_test, test_scores, threshold=0.5)
        results[method_name] = metrics
        print(
            f"  {method_name}: F1={metrics['f1']:.4f}  "
            f"Recall={metrics['recall']:.4f}  "
            f"ROC_AUC={metrics['roc_auc']:.4f}"
        )

    return results


def print_des_summary(des_results: Dict[str, Dict[str, Any]]) -> None:
    """Print a compact comparison table of DES methods."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        tbl = Table(title="Dynamic Ensemble Selection", header_style="bold cyan", show_lines=True)
        tbl.add_column("Method", style="bold")
        tbl.add_column("F1", justify="right")
        tbl.add_column("Recall", justify="right")
        tbl.add_column("Precision", justify="right")
        tbl.add_column("ROC AUC", justify="right")

        for name, m in des_results.items():
            tbl.add_row(
                name,
                f"{m['f1']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['precision']:.4f}",
                f"{m['roc_auc']:.4f}",
            )
        console.print(tbl)

    except ImportError:
        header = f"{'Method':12s} | {'F1':>8} | {'Recall':>8} | {'Prec':>8} | {'ROC AUC':>8}"
        print(header)
        print("-" * len(header))
        for name, m in des_results.items():
            print(
                f"{name:12s} | {m['f1']:>8.4f} | {m['recall']:>8.4f} | "
                f"{m['precision']:>8.4f} | {m['roc_auc']:>8.4f}"
            )
