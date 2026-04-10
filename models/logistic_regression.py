from __future__ import annotations

from sklearn.linear_model import LogisticRegression


def build_model(
    C: float = 1.0,
    penalty: str = "l2",
    class_weight: str = "balanced",
    random_state: int = 42,
    max_iter: int = 1000,
) -> LogisticRegression:

    return LogisticRegression(
        C=C,
        penalty=penalty,
        class_weight=class_weight,
        random_state=random_state,
        max_iter=max_iter,
        solver="lbfgs",
    )
