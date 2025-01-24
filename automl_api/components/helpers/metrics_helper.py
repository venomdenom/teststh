from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
)
from typing import Dict, List, Union
from enum import Enum


class MetricsCalculatorBase:
    """
    Base class for calculating metrics. Provides common logic for both classification and regression metrics.
    """
    metrics = {}

    @staticmethod
    def calculate(metrics_map: Dict, y_true, y_pred) -> Dict:
        """
        Generic method for calculating the given metrics.
        :param metrics_map: Dictionary {Enum: metric function}.
        :param y_true: Ground truth values.
        :param y_pred: Predicted values.
        :return: Dictionary of metrics with their calculated values.
        """
        results = {}
        for metric_enum, metric_func in metrics_map.items():
            results[metric_enum] = metric_func(y_true, y_pred)
        return results


class ClassificationMetrics(Enum):
    """
    Enum for classification metrics.
    """
    ACCURACY = 'accuracy'
    PRECISION = 'precision'
    RECALL = 'recall'
    F1_SCORE = 'f1_score'

    @classmethod
    def get_all_metrics(cls):
        """
        Returns a list of all available classification metrics.
        """
        return [cls.ACCURACY, cls.PRECISION, cls.RECALL, cls.F1_SCORE]


class ClassificationMetricsCalculator(MetricsCalculatorBase):
    """
    Calculator for classification metrics.
    """

    @staticmethod
    def calculate_metrics(y_true, y_pred) -> Dict[ClassificationMetrics, float]:
        """
        Calculate classification metrics based on true and predicted values.
        :param y_true: Ground truth values.
        :param y_pred: Predicted values.
        :return: Dictionary of calculated classification metrics.
        """
        metrics_map = {
            ClassificationMetrics.ACCURACY: accuracy_score,
            ClassificationMetrics.PRECISION: lambda y, p: precision_score(y, p, average='weighted', zero_division=0),
            ClassificationMetrics.RECALL: lambda y, p: recall_score(y, p, average='weighted', zero_division=0),
            ClassificationMetrics.F1_SCORE: lambda y, p: f1_score(y, p, average='weighted', zero_division=0),
        }
        return MetricsCalculatorBase.calculate(metrics_map, y_true, y_pred)


class RegressionMetrics(Enum):
    """
    Enum for regression metrics.
    """
    MSE = 'mean_squared_error'
    MAE = 'mean_absolute_error'
    MAPE = 'mean_absolute_percentage_error'
    R2 = 'r2_score'

    @classmethod
    def get_all_metrics(cls):
        """
        Returns a list of all available regression metrics.
        """
        return [cls.MSE, cls.MAE, cls.MAPE, cls.R2]


class RegressionMetricsCalculator(MetricsCalculatorBase):
    """
    Calculator for regression metrics.
    """

    @staticmethod
    def calculate_metrics(y_true, y_pred) -> Dict[RegressionMetrics, float]:
        """
        Calculate regression metrics based on true and predicted values.
        :param y_true: Ground truth values.
        :param y_pred: Predicted values.
        :return: Dictionary of calculated regression metrics.
        """
        metrics_map = {
            RegressionMetrics.MSE: mean_squared_error,
            RegressionMetrics.MAE: mean_absolute_error,
            RegressionMetrics.MAPE: mean_absolute_percentage_error,
            RegressionMetrics.R2: r2_score,
        }
        return MetricsCalculatorBase.calculate(metrics_map, y_true, y_pred)
