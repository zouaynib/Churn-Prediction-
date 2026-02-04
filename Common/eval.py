from __future__ import annotations 

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union   
from pathlib import Path

import json
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score, matthews_corrcoef, classification_report
from sklearn.exceptions import NotFittedError, UndefinedMetricWarning

def get_scores(model, X:np.ndarray) -> np.ndarray :

    if hasattr(model, "predict_proba") :
        return model.predict_proba(X)[:, 1]
    

    if hasattr(model, "decision_function"):
        s = np.asarray(model.decision_function(X), dtype=np.float64)
        den = (s.max() - s.min())
        if den > 0:
            s = (s - s.min()) / den
        else:
            s = np.zeros_like(s)
        return s


    return model.predict(X).astype(np.float64)



def predict_at_threshold(scores:np.ndarray, threshold:float) -> np.ndarray :

    return (scores >= threshold).astype(int) 

def tune_threshold(y_true:np.ndarray, scores:np.ndarray, metric: str ="f1", grid:Optional[np.ndarray]=None) -> Tuple[float, float] :

    if grid is None :
        grid = np.linspace(0.05, 0.95, 19)

    metric = metric.lower()
    best_t, best_v = 0.5, -1

    for t in grid :
        y_pred = predict_at_threshold(scores, float(t))

        if metric == "accuracy" :
            v = accuracy_score(y_true, y_pred, zero_division=0)
        elif metric == "precision" :
            v = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall" :
            v = recall_score(y_true, y_pred,zero_division=0)
        elif metric == "f1" :
            v = f1_score(y_true, y_pred,zero_division=0)
        elif metric == "mcc" :
            v = matthews_corrcoef(y_true, y_pred)
        else :
            raise ValueError(f"Unsupported metric: {metric}")

        if v > best_v :
            best_t, best_v = float(t), float(v)

    return best_t, best_v

def evaluate_binary(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
    prefix: str = ""
) -> Dict[str, Any]:

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    y_pred = predict_at_threshold(scores, threshold)

    out: Dict[str, Any] = {
        f"{prefix}threshold": float(threshold),
        f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}precision": float(precision_score(y_true, y_pred, zero_division=0)),
        f"{prefix}recall": float(recall_score(y_true, y_pred, zero_division=0)),
        f"{prefix}f1": float(f1_score(y_true, y_pred, zero_division=0)),
        f"{prefix}roc_auc": float(roc_auc_score(y_true, scores)),
        f"{prefix}pr_auc": float(average_precision_score(y_true, scores)),
        f"{prefix}confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        # helpful for your report/defense (not too big)
        f"{prefix}classification_report": classification_report(
            y_true, y_pred, digits=4, zero_division=0, output_dict=True
        ),
    }
    return out



def save_json(report: Dict[str, Any], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))


    """ customers who leave are what we are trying yo prevent that's why false negatives are more 
    expensive : predicting there i no churn why there actually is. that's I will lower  the threshold to raise the f1 score and the recall
    
    Recall : of all customers who actually churned, how much did the model catch ? TP / TP + FN"""




