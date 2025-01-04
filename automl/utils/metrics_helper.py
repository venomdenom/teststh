import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
)
from typing import Dict, List
from .data_helper import DataHelper

class ClassificationMetrics:
    def __init__(self, y_true: List, y_pred: List):
        self.y_true = y_true
        self.y_pred = y_pred
        self.metrics = None

    def calculate_metrics(self) -> Dict[str, float]:
        self.metrics = {
            "accuracy": accuracy_score(self.y_true, self.y_pred),
            "precision": precision_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
            "recall": recall_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        }

        return self.metrics

    def get_metric(self, key: str) -> float | None:
        return self.metrics.get(key, None)


class RegressionMetrics:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.metrics = None

    def calculate_metrics(self) -> Dict[str, float]:
        self.metrics = {
            "mean_squared_error": mean_squared_error(self.y_true, self.y_pred),
            "mean_absolute_error": mean_absolute_error(self.y_true, self.y_pred),
            'mean_absolute_percentage_error': mean_absolute_percentage_error(self.y_true, self.y_pred),
            "r2_score": r2_score(self.y_true, self.y_pred)
        }
        return self.metrics

    def get_metric(self, key: str) -> float | None:
        return self.metrics.get(key, None)


class MetricsRecommender:
    def __init__(self, y: List | np.ndarray) -> None:
        self.y = y
        self.target_type = None

    def suggest_metric(self) -> Dict[str, List[str]]:
        self.target_type = DataHelper.determine_target_type(self.y)
        if self.target_type == 'classification':
            return self._suggest_classification_metric()
        else:
            return self._suggest_regression_metric()

    def _suggest_classification_metric(self) -> Dict[str, List[str]]:
        best_metric = 'accuracy' if DataHelper.check_class_imbalance(self.y) else 'f1_score'
        other_metrics = ["precision", "recall", "f1_score"] if best_metric == "accuracy" else ["accuracy", "precision", "recall"]
        return {
            "best_metric": best_metric,
            "other_metrics": other_metrics
        }

    def _suggest_regression_metric(self) -> Dict[str, List[str]]:
        variance = np.var(self.y)
        if variance > 1:
            best_metric = "mean_squared_error"
        else:
            best_metric = "mean_absolute_error"
        other_metrics = ["r2_score", "mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error"]
        other_metrics.remove(best_metric)
        return {
            "best_metric": best_metric,
            "other_metrics": other_metrics
        }
