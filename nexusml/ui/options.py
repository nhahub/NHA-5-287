"""Choices shared by the new-analysis composer and the retry dialog."""
from typing import Any, Dict, Optional

PROBLEMS = {"Decide for me": None, "Predict a category (classification)": "classification", "Predict a number (regression)": "regression", "Forecast over time": "time_series",
            "Find natural groups (segmentation)": "clustering", "Describe only, no model": "descriptive"}
METRICS = {"No target": None, "AUC": "roc_auc", "F1": "f1", "Accuracy": "accuracy", "Recall": "recall", "Precision": "precision", "R²": "r2", "RMSE (lower is better)": "rmse", "MAE (lower is better)": "mae"}


def build_spec(metric: Optional[str], value: float, second: Optional[str], second_value: float) -> Optional[Dict[str, Any]]:
    if not metric:
        return None
    spec: Dict[str, Any] = {"metric": metric, ("max" if metric in ("rmse", "mae") else "min"): float(value)}
    if second and second != metric:
        spec["subject_to"] = {"metric": second, ("max" if second in ("rmse", "mae") else "min"): float(second_value)}
    return spec
