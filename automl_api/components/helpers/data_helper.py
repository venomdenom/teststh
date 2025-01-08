import numpy as np
from sklearn.utils.multiclass import type_of_target
from collections import Counter
from typing import List
from base_helper import TaskType

class DataHelper:

    @staticmethod
    def get_task_type(y: List | np.ndarray) -> TaskType:
        target_type = type_of_target(y)
        if target_type in ['binary', 'multiclass']:
            return TaskType.CLASSIFICATION
        elif target_type in ['continuous', 'continuous-multioutput']:
            return TaskType.REGRESSION
        else:
            raise ValueError(f"Unsupported target type: {target_type}")

    @staticmethod
    def check_class_imbalance(y: List | np.ndarray, threshold: float = 0.7) -> bool:
        class_counts = Counter(y)
        total_samples = sum(class_counts.values())
        max_class_ratio = max(class_counts.values()) / total_samples

        return max_class_ratio > threshold

    @staticmethod
    def calculate_variance(y: List | np.ndarray) -> float:
        return float(np.var(y))
