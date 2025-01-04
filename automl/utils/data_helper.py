import numpy as np
from sklearn.utils.multiclass import type_of_target
from collections import Counter
from typing import List


class DataHelper:

    @staticmethod
    def determine_target_type(y: List | np.ndarray) -> str:
        target_type = type_of_target(y)
        if target_type in ['binary', 'multiclass']:
            return 'classification'
        elif target_type in ['continuous', 'continuous-multioutput']:
            return 'regression'
        else:
            raise ValueError(f"Unsupported target type: {target_type}")

    @staticmethod
    def check_class_imbalance(y: List | np.ndarray, threshold: float = 0.7) -> bool:
        class_counts = Counter(y)
        total_samples = sum(class_counts.values())
        max_class_ratio = max(class_counts.values()) / total_samples

        return max_class_ratio > threshold
