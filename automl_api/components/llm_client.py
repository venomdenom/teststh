from typing import Optional, Dict
import pandas as pd
import requests
from automl_api.components.helpers.base_helper import TaskType
from automl_api.components.helpers.metrics_helper import ClassificationMetrics, RegressionMetrics


type Metric = ClassificationMetrics | RegressionMetrics

class LLMClient:
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

    def _send_request(self, prompt: str, max_tokens: int = 100) -> Optional[Dict]:
        """
        Send a request to the LLM API.
        :param prompt: text prompt to send.
        :param max_tokens: maximum number of tokens to send.
        :return: response from LLM API with type Dict or None.
        """
        payload = {
            'prompt': prompt,
            'max_tokens': max_tokens
        }

        try:
            response = requests.post(self.api_url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f'Failed to send request to LLM API: {e}')

    def analyze_users_input(self, user_prompt: str) -> str:
        """
        Analyze user input. If the user input is non-informative, returns modified prompt.
        :param user_prompt: text prompt to send.
        :return: modified prompt.
        """
        prompt = (
            f"Analyze the following user input and determine whether it contains clear goals or preferences for a machine learning task. "
            f"If the input is vague or incomplete, suggest a more detailed interpretation that could guide model and metric selection. "
            f"Here is the input: \"{user_prompt}\""
        )
        response = self._send_request(prompt, max_tokens=150)
        analysis = response.get("choices", [{}])[0].get("text", "").strip()

        if not analysis:
            return "The user input is unclear. Defaulting to general recommendations."

        return analysis

    def generate_dataset_description(self, data: pd.DataFrame) -> str:
        """
        Generate dataset description based on its structure.
        :param data: pandas DataFrame containing dataset.
        :return: dataset description.
        """
        columns = ", ".join(data.columns.tolist())
        sample_rows = data.head(10).to_dict(orient='records')
        sample_text = "\n".join([str(row) for row in sample_rows])

        prompt = (
            f"The dataset contains the following columns: {columns}. "
            f"Here are the first 10 rows of the dataset:\n{sample_text}\n"
            f"Based on this data, describe the dataset."
        )
        response = self._send_request(prompt, max_tokens=200)
        description = response.get("choices", [{}])[0].get("text", "").strip()

        return description

    def determine_task_type(self, description: str, user_prompt: Optional[str] = None) -> TaskType:
        """
        Determine the task type based on the description.
        :param description: description of the dataset.
        :param user_prompt: user prompt for the task.
        :return: TaskType.
        """
        prompt = (
            f"Based on the following dataset description and user input, determine the task type "
            f"(classification, regression, or clustering)."
            f"\nDataset description: {description}\n"
        )
        if user_prompt:
            prompt += f"User input: {user_prompt}\n"

        response = self._send_request(prompt)
        task_prediction = response.get("choices", [{}])[0].get("text", "").strip().lower()
        if "classification" in task_prediction:
            return TaskType.CLASSIFICATION
        elif "regression" in task_prediction:
            return TaskType.REGRESSION
        elif "clustering" in task_prediction:
            return TaskType.CLUSTERING
        else:
            raise ValueError(f"LLM could not determine task type: {task_prediction}")

    def get_recommended_metric(self, user_prompt: str, dataset_prompt: str, task_type: TaskType) -> Metric:
        """
        Recommend a metric based on user input and task type.
        :param user_prompt: User input describing preferences for the evaluation metric.
        :param task_type: Task type (classification or regression).
        :return: Recommended metric as an instance of the appropriate Metric Enum class.
        """
        prompt = (
            f"The user is working on a {task_type.name.lower()} task and provided the following input: '{user_prompt}'. "
            f"Recommend the most suitable evaluation metric for this task. "
            f"\nDataset description: {dataset_prompt}\n"
            f"For classification, consider metrics like accuracy, precision, recall, and f1_score. "
            f"For regression, consider metrics like mse, mae, mape, and r2_score."
        )

        response = self._send_request(prompt, max_tokens=150)
        recommended_metric = response.get("choices", [{}])[0].get("text", "").strip().lower()

        if task_type == TaskType.CLASSIFICATION:
            valid_metrics = {metric.value: metric for metric in ClassificationMetrics}
        elif task_type == TaskType.REGRESSION:
            valid_metrics = {metric.value: metric for metric in RegressionMetrics}
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        if recommended_metric in valid_metrics:
            return valid_metrics[recommended_metric]
        else:
            raise ValueError(f"Invalid metric recommended by LLM: {recommended_metric}. "
                             f"Valid metrics are: {[metric.value for metric in valid_metrics.values()]}")