from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error, f1_score, r2_score
from typing import Any, Dict, Tuple
import numpy as np
from .metrics_helper import MetricsRecommender


class ModelSelector:
    def __init__(self, X: np.ndarray, y: np.ndarray, task_type: str = None):
        """
        Initialize the ModelSelector.

        Args:
            X (np.ndarray): Features matrix.
            y (np.ndarray): Target vector.
            task_type (str): 'classification' or 'regression'. If None, it will be inferred.
        """
        self.X = X
        self.y = y
        self.task_type = task_type or MetricsRecommender(y).target_type
        self.models = self._get_models()
        self.best_model = None
        self.best_score = None

    def _get_models(self) -> Dict[str, Any]:
        """
        Get a dictionary of models based on task type.

        Returns:
            Dict[str, Any]: Models for classification or regression.
        """
        if self.task_type == 'classification':
            return {
                "LogisticRegression": LogisticRegression(),
                "RandomForestClassifier": RandomForestClassifier(),
                "GradientBoostingClassifier": GradientBoostingClassifier()
            }
        elif self.task_type == 'regression':
            return {
                "LinearRegression": LinearRegression(),
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor()
            }
        else:
            raise ValueError("Unsupported task type. Must be 'classification' or 'regression'.")

    def select_model(self, metric: str = None, test_size: float = 0.2, random_state: int = 42) -> Tuple[Any, float]:
        """
        Select the best model based on the specified metric.

        Args:
            metric (str): Metric to use for evaluation. Defaults to recommended metric.
            test_size (float): Test size for splitting the data.
            random_state (int): Random state for reproducibility.

        Returns:
            Tuple[Any, float]: Best model and its score.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

        # Determine metric
        metric = metric or MetricsRecommender(self.y).suggest_metric()["best_metric"]
        scoring = self._get_scorer(metric)

        # Train and evaluate models
        for model_name, model in self.models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            score = scoring(y_test, predictions)
            if self.best_score is None or score > self.best_score:
                self.best_model = model
                self.best_score = score

        return self.best_model, self.best_score

    def _get_scorer(self, metric: str):
        """
        Get the scoring function based on the metric.

        Args:
            metric (str): Metric name.

        Returns:
            Callable: Scoring function.
        """
        if metric == "accuracy":
            return accuracy_score
        elif metric == "f1_score":
            return lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted', zero_division=0)
        elif metric == "mean_squared_error":
            return lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)  # Negative for maximization
        elif metric == "r2_score":
            return r2_score
        else:
            raise ValueError(f"Unsupported metric: {metric}")
