from __future__ import annotations
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

def build_model_bagging(
    n_estimators: int = 300,
    max_samples: float = 0.8,
    max_features: float = 1.0,
    base_max_depth: int | None = None,
    random_state: int = 42,
) -> BaggingClassifier:
    base = DecisionTreeClassifier(max_depth=base_max_depth, random_state=random_state)
    return BaggingClassifier(
        estimator=base,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        n_jobs=-1,
        random_state=random_state,
    )
