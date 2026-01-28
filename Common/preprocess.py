from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from Common.data import read_csv

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


#data containers

@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


@dataclass
class PreprocessingOutput:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    preprocessor: ColumnTransformer




def load_data(csv_path: str, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    data = read_csv(Path(csv_path))


    if "CustomerID" in data.columns:
        data = data.drop(columns=["CustomerID"])


    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")


    if target_col not in data.columns:
        raise ValueError(f"Expected '{target_col}' not found in data")

    churn_map = {"Yes": 1, "No": 0, "1": 1, "0": 0, 1: 1, 0: 0}
    data[target_col] = data[target_col].map(churn_map)

    n_before = len(data)
    data = data.dropna(subset=[target_col])
    dropped = n_before - len(data)
    if dropped > 0:
        print(f"[INFO] Dropped {dropped} rows with missing target")

    data[target_col] = data[target_col].astype(int)

    data = data.drop_duplicates()

    X = data.drop(columns=[target_col])
    y = data[target_col]

    return X, y



def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True
) -> SplitData:

    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0")

    strat = y if stratify else None


    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(test_size + val_size),
        random_state=random_state,
        stratify=strat
    )

    # No validation set
    if val_size == 0:
        return SplitData(
            X_train=X_train,
            X_val=X_temp.iloc[:0].copy(),
            X_test=X_temp,
            y_train=y_train.to_numpy(),
            y_val=y_temp.iloc[:0].to_numpy(),
            y_test=y_temp.to_numpy()
        )

    # Split temp → val / test
    test_frac_of_temp = test_size / (test_size + val_size)
    strat2 = y_temp if stratify else None

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_frac_of_temp,
        random_state=random_state,
        stratify=strat2
    )

    return SplitData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train.to_numpy(),
        y_val=y_val.to_numpy(),
        y_test=y_test.to_numpy()
    )


# ============================================================
# Preprocessor (learned ONLY on train)
# ============================================================

def build_preprocessor(
    X_train: pd.DataFrame,
    scale_numeric: bool = False,
    scaler_type: str = "standard"
) -> tuple[ColumnTransformer, List[str], List[str]]:

    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    num_steps = [("imputer", SimpleImputer(strategy="median"))]

    if scale_numeric:
        if scaler_type == "standard":
            num_steps.append(("scaler", StandardScaler()))
        elif scaler_type == "minmax":
            num_steps.append(("scaler", MinMaxScaler()))
        else:
            raise ValueError(f"Unsupported scaler_type: {scaler_type}")

    numeric_pipe = Pipeline(num_steps)

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )

    return preprocessor, numeric_cols, categorical_cols


def get_feature_names(
    preprocessor: ColumnTransformer,
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> List[str]:

    feature_names = list(numeric_cols)
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    feature_names.extend(ohe.get_feature_names_out(categorical_cols))
    return feature_names


# ============================================================
# End-to-end preprocessing
# ============================================================

def preprocess_data(
    csv_path: str,
    target_col: str = "Churn",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True,
    scale_numeric: bool = False,
    scaler_type: str = "standard"
) -> PreprocessingOutput:

    X, y = load_data(csv_path, target_col)

    splits = split_train_val_test(
        X, y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        stratify=stratify
    )

    preprocessor, num_cols, cat_cols = build_preprocessor(
        splits.X_train,
        scale_numeric=scale_numeric,
        scaler_type=scaler_type
    )

    X_train_p = preprocessor.fit_transform(splits.X_train)
    X_val_p = preprocessor.transform(splits.X_val) if len(splits.X_val) > 0 else np.empty((0, X_train_p.shape[1]))
    X_test_p = preprocessor.transform(splits.X_test)

    feature_names = get_feature_names(preprocessor, num_cols, cat_cols)

    return PreprocessingOutput(
        X_train=X_train_p,
        X_val=X_val_p,
        X_test=X_test_p,
        y_train=splits.y_train,
        y_val=splits.y_val,
        y_test=splits.y_test,
        feature_names=feature_names,
        preprocessor=preprocessor
    )




if __name__ == "__main__":
    out = preprocess_data("Data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print("Shapes:", out.X_train.shape, out.X_val.shape, out.X_test.shape)
    print("Churn rate:", out.y_train.mean(), out.y_val.mean(), out.y_test.mean())
    print("Features:", len(out.feature_names))
