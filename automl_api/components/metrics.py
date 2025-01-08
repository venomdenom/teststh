import numpy as np
from typing import Dict, List
from .helpers.data_helper import DataHelper
from .helpers.base_helper import TaskType
from .helpers.metrics_helper import ClassificationMetrics, RegressionMetrics

type Metrics = ClassificationMetrics | RegressionMetrics

class MetricsRecommender:
    def __init__(self, target: np.ndarray) -> None:
        self.target = target

    def suggest_metric(self) -> Dict[str, List[Metrics]]:
        task_type = DataHelper.get_task_type(self.target)
        if task_type == TaskType.CLASSIFICATION:
            return self._suggest_classification_metric()
        elif task_type == TaskType.REGRESSION:
            return self._suggest_regression_metric()

    def _suggest_classification_metric(self) -> Dict[str, List[ClassificationMetrics]]:
        class_imbalance = DataHelper.check_class_imbalance(self.target)
        best_metric = ClassificationMetrics.ACCURACY if class_imbalance else ClassificationMetrics.F1SCE
        other_metrics = ClassificationMetrics.get_all_metrics().remove(best_metric)

        return {
            'best_metric': best_metric,
            'other_metrics': other_metrics,
        }

    def _suggest_regression_metric(self):
        variance = DataHelper.calculate_variance(self.target)
        if variance > 1:
            best_metric = RegressionMetrics.MSE
        else:
            best_metric = RegressionMetrics.MAE

        other_metrics = RegressionMetrics.get_all_metrics().remove(best_metric)

        return {
            'best_metric': best_metric,
            'other_metrics': other_metrics,
        }