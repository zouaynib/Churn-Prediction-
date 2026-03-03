from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier


def build_model(
    n_estimators: int = 300,
    learning_rate: float = 0.03,
    max_depth: int = 3,
    subsample: float = 0.8,
    random_state: int = 42,
    min_samples_split = 20,
    class_weight="balanced"
) -> GradientBoostingClassifier:

    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=random_state,
    )
