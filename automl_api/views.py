import os

import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from django.core.files.storage import FileSystemStorage
from automl_api.components.preprocessing import DataPreprocessor
from automl_api.components.metrics import MetricsRecommender
from automl_api.components.helpers.metrics_helper import ClassificationMetrics, RegressionMetrics
from automl_api.components.helpers.model_helper import ModelSelector
from automl_api.components.helpers.visualizer_helper import VisualizerHelper


class UploadDataset(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        file = request.FILES.get('file')
        if not file:
            return Response({'error': 'Файл не предоставлен'}, status=400)

        # Saving file into temporary directory
        fs = FileSystemStorage(location='/tmp/')
        filename = fs.save(file.name, file)
        file_path = fs.path(filename)

        try:
            # DataPreprocessor initialization
            preprocessor = DataPreprocessor(file_path)
            preprocessor.load_data()
            preprocessor.preprocess(
                fill_with_median=True,
                scaler_type='minmax'
            )

            # Saving preprocessed data
            processed_file_path = f"/tmp/processed_{file.name}"
            preprocessor.data.to_csv(processed_file_path, index=False)

            return Response({
                'message': 'Данные успешно обработаны!',
                'processed_file_path': processed_file_path
            })
        except Exception as e:
            return Response({'error': f'Ошибка при обработке данных: {str(e)}'}, status=500)


class CalculateMetrics(APIView):
    def post(self, request):
        y_true = request.data.get("y_true")
        y_pred = request.data.get("y_pred")

        if not y_true or not y_pred:
            return Response({"error": "Both y_true and y_pred must be provided."}, status=400)

        metric_type = request.data.get("metric_type", "classification")  # classification or regression
        if metric_type == "classification":
            metrics = ClassificationMetrics(y_true, y_pred)
        elif metric_type == "regression":
            metrics = RegressionMetrics(y_true, y_pred)
        else:
            return Response({"error": f"Unsupported metric type: {metric_type}"}, status=400)

        calculated_metrics = metrics.calculate_metrics()
        return Response({"metrics": calculated_metrics}, status=200)

class RecommendMetric(APIView):
    def post(self, request):
        y = request.data.get("y")
        if not y:
            return Response({"error": "Target variable (y) must be provided."}, status=400)

        recommender = MetricsRecommender(y)
        recommendations = recommender.suggest_metric()
        return Response({"recommendations": recommendations}, status=200)


class SelectModel(APIView):
    def post(self, request):
        X = request.data.get("X")
        y = request.data.get("y")
        task_type = request.data.get("task_type")  # classification or regression

        if not X or not y:
            return Response({"error": "X and y must be provided."}, status=400)

        selector = ModelSelector(X, y, task_type)
        best_model, best_score = selector.select_model()
        return Response({"best_model": str(best_model), "best_score": best_score}, status=200)

class VisualizeData(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        file_path = request.data.get("file_path")
        plot_type = request.data.get("plot_type")
        title = request.data.get("title")
        column = request.data.get("column")
        x = request.data.get("x")
        y = request.data.get("y")

        if not file_path or not os.path.exists(file_path):
            return Response({"error": "No file provided."}, status=400)

        try:
            data = pd.read_csv(file_path)
        except Exception as e:
            return Response({"error": f"Error while reading file - {str(e)}."}, status=400)

        visualizer = VisualizerHelper()
        plot_path = None

        try:
            match plot_type:
                case "histogram":
                    if not column:
                        return Response({"error": "Column must be provided."}, status=400)
                    plot_path = visualizer.plot_histogram(data, column=column, title=title)
                case "scatter":
                    if not (x and y):
                        return Response({"error": "Both x and y must be provided."}, status=400)
                    plot_path = visualizer.plot_scatter(data, x, y, title=title)
                case "correlation_matrix":
                    plot_path = visualizer.plot_correlation_matrix(data)
                case _:
                    return Response({"error": f"Unsupported plot type: {plot_type}"}, status=400)
        except Exception as e:
            return Response({"error": f"Error while plotting data - {str(e)}."}, status=400)

        return Response({"plot_path": plot_path}, status=200)