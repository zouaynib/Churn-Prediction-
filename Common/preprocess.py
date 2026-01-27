from Common import data
import numpy as np 
import pandas as pd 


from _future_ import annotations
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.pipeline import Pipeline
from dataclasses import dataclass   


@dataclass
class   SplitData : 
    X_train : pd.DataFrame
    X_val : pd.DataFrame 
    X_test : pd.DataFrame
    y_train : pd.Series
    y_val : pd.Series
    y_test : pd.Series


def load_data(csv_path: str, target_col: str) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    
    if "CustomerID" in data.columns :
        data = data.drop(columns=["CustomerID"])
        
    if "Churn" not in data.columns :
        raise ValueError("Expected 'Churn' not found in data")
    
    churn_map = {"Yes" : 1, "No" : 0, "1" : 1, "0" : 0, 1 : 1, 0 : 0}
    data["Churn"] = data["Churn"].map(churn_map)

    data = data.drop_duplicates()

    if "TotalCharges" in data.columns :
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors='coerce')
    
    X = data.drop(columns=[target_col])
    y = data[target_col]

    return data


def split_train_val_test(
    data: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    val_size : float = 0.1,
    random_state: int=42,
    stratify : bool =True )-> SplitData :

    X_train, X_temp, y_train, y_temp = train_test_split( X, y, test_size = (test_size + val_size), random_state=random_state, stratify=stratify)

    if val_size == 0:
        
        X_val = X_temp.iloc[:0].copy()
        y_val = y_temp.iloc[:0].to_numpy()
        X_test = X_temp
        y_test = y_temp.to_numpy()
        return SplitData(X_train, X_val, X_test, y_train.to_numpy(), y_val, y_test)
    
    relative_val_size = val_size / test_size + val_size

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size= relative_val_size, random_state=random_state, stratify=stratify)

    return SplitData(X_train = X_train, X_val = X_val, X_test = X_test, y_train = y_train.to_numpy(), y_val = y_val.to_numpy(), y_test = y_test.to_numpy() )


def scale_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_type: str = "standard"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    if scaler_type == "standard":
        scaler = preprocessing.StandardScaler()
    elif scaler_type == "minmax":
        scaler = preprocessing.MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaler type: {scaler_type}")

    scaler.fit(X_train)

    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    return X_train_scaled, X_val_scaled, X_test_scaled  


def preprocess_data( 
    csv_path: str,
    test_size: float= 0.2,
    val_size : float= 0.1,
    random_state: int = 42,
    stratify: bool = True,
    scale_numeric : bool = True ) -> PreprocessingOutput :

    data = load_data(csv_path)
    X = data.drop(columns=["Churn"])
    y = data["Churn"]

    splits = split_train_val_test(X, y, test_size=test_size, val_size= val_size, random_state = random_state, stratify = stratify)

