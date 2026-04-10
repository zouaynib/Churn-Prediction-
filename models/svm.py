from __future__ import annotations

from sklearn.svm import SVC


def build_model(
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str = "scale",
    class_weight: str = "balanced",
    random_state: int = 42,
) -> SVC:

    return SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        class_weight=class_weight,
        probability=True,
        random_state=random_state,
    )
