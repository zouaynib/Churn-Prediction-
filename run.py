import numpy as np
from sklearn.pipeline import Pipeline

from Common.config import (
    DATA_PATH, DEFAULT_THRESHOLD, SCALE_NUMERIC, SCALER_TYPE, STRATIFY,
    TARGET_COL, TEST_SIZE, VAL_SIZE,
)
from Common.eval import evaluate_binary, tune_threshold
from Common.preprocess import preprocess_data
from models.adaboost import build_model_adaboost


def main() -> None:
    data = preprocess_data(
        DATA_PATH,
        target_col=TARGET_COL,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        stratify=STRATIFY,
        scale_numeric=SCALE_NUMERIC,
        scaler_type=SCALER_TYPE,
    )

    model = build_model_adaboost()
    pipeline = Pipeline([("model", model)])
    pipeline.fit(data.X_train, data.y_train)

    grid = np.linspace(0.001, 0.999, 999)

    # Validation scores
    val_scores = pipeline.predict_proba(data.X_val)[:, 1]

    # Tune thresholds
    t_f1, v_f1 = tune_threshold(data.y_val, val_scores, metric="f1", grid=grid)
    t_mcc, v_mcc = tune_threshold(data.y_val, val_scores, metric="mcc", grid=grid)

    print(f"[VAL] Best F1  threshold = {t_f1:.3f} | F1  = {v_f1:.4f}")
    print(f"[VAL] Best MCC threshold = {t_mcc:.3f} | MCC = {v_mcc:.4f}")

    # Test scores
    test_scores = pipeline.predict_proba(data.X_test)[:, 1]

    # Evaluate on test for each threshold (plus default)
    rep_default = evaluate_binary(data.y_test, test_scores, threshold=t_mcc, prefix="test_default_")
    rep_f1 = evaluate_binary(data.y_test, test_scores, threshold=t_f1, prefix="test_f1_")
    rep_mcc = evaluate_binary(data.y_test, test_scores, threshold=t_mcc, prefix="test_mcc_")

    print(rep_default)
    print(rep_f1)
    print(rep_mcc)


if __name__ == "__main__":
    main()
