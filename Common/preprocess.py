from __future__ import annotations

from Common import data
import numpy as np 
import pandas as pd 



from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from skelearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer 



from dataclasses import dataclass  
from typing import List,Tuple, Optional




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

    if "TotalCharges" in data.columns :
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors='coerce')

    if target_col not in data.columns :
        raise ValueError("Expected {target_col} not found in data")
    
    churn_map = {"Yes" : 1, "No" : 0, "1" : 1, "0" : 0, 1 : 1, 0 : 0}
    data[target_col] = data[target_col].map(churn_map)

    n_before = len(data)
    data = data.dropna(subset=[target_col])
    dropped = n_before - len(data)
    if dropped > 0 :
        print(f"Dropped {dropped} rows with missing target values")

    data[target_col] = data[target_col].astype(int)

    data = data.drop_duplicates()

    X= data.drop(columns=[target_col])
    y= data[target_col]


    return X, y

def split_train_val_test(
    data: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    val_size : float = 0.1,
    random_state: int=42,
    stratify : bool =True )-> SplitData :

    if test_size + val_size >= 1.0 :
        raise ValueError("Test size and validation size must sum to less than 1.0") 
    
    stratify = y if stratify else None

    

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

    data = load_data(csv_path, target_col="Churn")
    X = data.drop(columns=["Churn"])
    y = data["Churn"]

    splits = split_train_val_test(X, y, test_size=test_size, val_size= val_size, random_state = random_state, stratify = stratify)

