from __future__ import annotations

import lightgbm as lgb


def build_model(
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = -1,
    num_leaves: int = 31,
    subsample: float = 0.8,
    subsample_freq: int = 1,
    colsample_bytree: float = 0.8,
    min_child_samples: int = 20,
    reg_alpha: float = 0.1,
    reg_lambda: float = 0.1,
    random_state: int = 42,
    class_weight: str | dict | None = "balanced",
    verbose: int = -1,
) -> lgb.LGBMClassifier:

    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        subsample=subsample,
        subsample_freq=subsample_freq,
        colsample_bytree=colsample_bytree,
        min_child_samples=min_child_samples,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        class_weight=class_weight,
        verbose=verbose,
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
    )