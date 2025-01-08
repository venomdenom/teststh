from typing import Dict, List
import numpy as np
from sklearn.model_selection import GridSearchCV
from .helpers.data_helper import DataHelper
from .helpers.metrics_helper import ClassificationMetrics, RegressionMetrics
from .helpers.model_helper import ModelsConfig


type Metric = ClassificationMetrics | RegressionMetrics
class ModelSelector:
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def select_model(self, metrics: Dict[str, Metric | List[Metric]]):
        best_models = {}
        task_type = DataHelper.get_task_type(self.y_train)
        models = ModelsConfig.config.get(task_type)

        if models is None:
            raise ValueError("No models configured")
        else:
            for model_enum, model_info in models.items():
                model_class = model_info['class']
                param_grid = model_info['params']

                grid_search = GridSearchCV(estimator=model_class(), param_grid=param_grid, cv=5, scoring=metrics['best_metric'].value)
                grid_search.fit(self.X_train, self.y_train)

                best_models[model_enum] = {
                    'best_estimator': grid_search.best_estimator_,
                    'best_params': grid_search.best_params_,
                    'metrics': grid_search.best_score_
                }

        return max(best_models.items(), key=lambda item: item[1]['metrics'])