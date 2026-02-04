from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier


def build_model(
    n_estimators: int = 800,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str | float = "sqrt",
    class_weight: str | dict | None = "balanced",
    random_state: int = 42,
) -> ExtraTreesClassifier:

    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=random_state,
    )
