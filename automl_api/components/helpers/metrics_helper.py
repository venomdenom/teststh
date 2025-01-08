from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
)
from typing import Dict
from enum import Enum


class ClassificationMetrics(Enum):
    ACCURACY = 'accuracy'
    PRECISION = 'precision'
    RECALL = 'recall'
    F1SCE = 'f_1'

    @classmethod
    def get_all_metrics(cls):
        return [cls.ACCURACY, cls.PRECISION, cls.RECALL, cls.F1SCE]


class ClassificationMetricsCalculator:
    def __init__(self):
        self.metrics = None

    @classmethod
    def calculate_metrics(cls, y_true, y_pred) -> Dict[ClassificationMetrics, float | int]:
        cls.metrics = {
            ClassificationMetrics.ACCURACY: accuracy_score(y_true, y_pred),
            ClassificationMetrics.PRECISION: precision_score(y_true, y_pred, average='weighted', zero_division=0),
            ClassificationMetrics.RECALL: recall_score(y_true, y_pred, average='weighted', zero_division=0),
            ClassificationMetrics.F1SCE: f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

        return cls.metrics

    def get_metric(self, key: str) -> float | None:
        return self.metrics.get(key, None)


class RegressionMetrics(Enum):
    MSE = 'mean_squared_error'
    MAE = 'mean_absolute_error'
    MAPE = 'mean_absolute_percentage_error'
    R2 = 'r2_score'

    @classmethod
    def get_all_metrics(cls):
        return [cls.MSE, cls.MAE, cls.MAPE, cls.R2]

class RegressionMetricsCalculator:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.metrics = None

    def calculate_metrics(self) -> Dict[RegressionMetrics, float| int]:
        self.metrics = {
            RegressionMetrics.MSE: mean_squared_error(self.y_true, self.y_pred),
            RegressionMetrics.MAE: mean_absolute_error(self.y_true, self.y_pred),
            RegressionMetrics.MAPE: mean_absolute_percentage_error(self.y_true, self.y_pred),
            RegressionMetrics.R2: r2_score(self.y_true, self.y_pred)
        }

        return self.metrics

    def get_metric(self, key: str) -> float | None:
        return self.metrics.get(key, None)