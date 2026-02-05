from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb


def build_model(
    use_probabilities: bool = True,
    final_estimator_C: float = 1.0,
    cv: int = 5,
    n_jobs: int = -1,
    passthrough: bool = False,
) -> StackingClassifier:

    
    base_learners = [
        (
            "lightgbm",
            lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=0.1,
                class_weight="balanced",
                random_state=42,
                verbose=-1,
            ),
        ),
        (
            "xgboost",
            xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=3,  
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
            ),
        ),
        (
            "gradient_boosting",
            GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
    
    meta_learner = LogisticRegression(
        C=final_estimator_C,
        class_weight="balanced",
        random_state=42,
        max_iter=1000,
        solver="lbfgs",
    )
    
    return StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=cv,
        stack_method="predict_proba" if use_probabilities else "predict",
        n_jobs=n_jobs,
        passthrough=passthrough,
        verbose=0,
    )