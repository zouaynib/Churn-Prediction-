from __future__ import annotations

def build_model(
    n_estimators: int = 800,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    scale_pos_weight: float | None = None,  # set if you want cost-sensitive learning
    random_state: int = 42,
):

    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise ImportError(
            "xgboost is not installed. Install it with: pip install xgboost"
        ) from e

    params = dict(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        min_child_weight=min_child_weight,
        gamma=gamma,
        objective="binary:logistic",
        eval_metric="logloss",   # avoids warnings and is standard for binary
        n_jobs=-1,
        random_state=random_state,
        tree_method="hist",      # fast on CPU; use "gpu_hist" if you have CUDA
    )

    if scale_pos_weight is not None:
        params["scale_pos_weight"] = float(scale_pos_weight)

    return XGBClassifier(**params)
