from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from typing import Dict
from enum import Enum
from automl_api.components.helpers.base_helper import TaskType
from scipy.stats import uniform, randint

from automl_api.components.metrics import MetricsRecommender


class ClassificationModels(Enum):
    LogisticRegression = 1
    RandomForestClassifier = 2
    GradientBoostingClassifier = 3


class RegressionModels(Enum):
    LinearRegression = 1
    RandomForestRegressor = 2
    GradientBoostingRegressor = 3


class ModelsConfig:
    Model = ClassificationModels | RegressionModels
    config: Dict[TaskType, Dict[Model, Dict]] = {
        TaskType.CLASSIFICATION: {
            ClassificationModels.LogisticRegression: {
                'class': LogisticRegression,
                'params': {
                    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                    'C': uniform(0.01, 10.0),  # Uniform distribution for C between 0.01 and 10.0
                    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                    'max_iter': randint(100, 500),  # Random integer between 100 and 500
                }
            },
            ClassificationModels.RandomForestClassifier: {
                'class': RandomForestClassifier,
                'params': {
                    'n_estimators': randint(100, 500),  # Random integer between 100 and 500
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': randint(2, 11),  # Random integer between 2 and 10
                    'min_samples_leaf': randint(1, 5),  # Random integer between 1 and 4
                    'bootstrap': [True, False],
                }
            },
            ClassificationModels.GradientBoostingClassifier: {
                'class': GradientBoostingClassifier,
                'params': {
                    'n_estimators': randint(100, 500),  # Random integer between 100 and 500
                    'learning_rate': uniform(0.01, 0.2),  # Uniform distribution for learning rate between 0.01 and 0.21
                    'max_depth': randint(3, 8),  # Random integer between 3 and 7
                    'subsample': uniform(0.8, 0.2),  # Uniform distribution between 0.8 and 1.0
                    'min_samples_split': randint(2, 11),  # Random integer between 2 and 10
                }
            }
        },
        TaskType.REGRESSION: {
            RegressionModels.LinearRegression: {
                'class': LinearRegression,
                'params': {
                    'fit_intercept': [True, False],
                    'normalize': [True, False],
                }
            },
            RegressionModels.RandomForestRegressor: {
                'class': RandomForestRegressor,
                'params': {
                    'n_estimators': randint(100, 500),  # Random integer between 100 and 500
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': randint(2, 11),  # Random integer between 2 and 10
                    'min_samples_leaf': randint(1, 5),  # Random integer between 1 and 4
                    'bootstrap': [True, False],
                }
            },
            RegressionModels.GradientBoostingRegressor: {
                'class': GradientBoostingRegressor,
                'params': {
                    'n_estimators': randint(100, 500),  # Random integer between 100 and 500
                    'learning_rate': uniform(0.01, 0.2),  # Uniform distribution for learning rate between 0.01 and 0.21
                    'max_depth': randint(3, 8),  # Random integer between 3 and 7
                    'subsample': uniform(0.8, 0.2),  # Uniform distribution between 0.8 and 1.0
                    'min_samples_split': randint(2, 11),  # Random integer between 2 and 10
                }
            }
        }
    }

    @classmethod
    def get_models_config(cls, task_type: TaskType) -> Dict[Model, Dict]:
        return cls.config.get(task_type, None)