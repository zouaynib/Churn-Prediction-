from sklearn.pipeline import Pipeline

from Common.config import (
    DATA_PATH,
    DEFAULT_THRESHOLD,
    SCALE_NUMERIC,
    SCALER_TYPE,
    STRATIFY,
    TARGET_COL,
    TEST_SIZE,
    VAL_SIZE,
)
from Common.eval import evaluate_binary
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
    pipeline = Pipeline(
        [
            ("model", model),
        ]
    )

    pipeline.fit(data.X_train, data.y_train)
    scores = pipeline.predict_proba(data.X_test)[:, 1]
    report = evaluate_binary(data.y_test, scores, threshold=DEFAULT_THRESHOLD)
    print(report)


if __name__ == "__main__":
    main()
