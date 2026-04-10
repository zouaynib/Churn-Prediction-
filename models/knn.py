from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier


def build_model(
    n_neighbors: int = 15,
    weights: str = "distance",
    metric: str = "minkowski",
    n_jobs: int = -1,
) -> KNeighborsClassifier:

    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        n_jobs=n_jobs,
    )
