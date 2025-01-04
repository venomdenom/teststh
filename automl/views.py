from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from django.core.files.storage import FileSystemStorage
from .utils.preprocessing_helper import DataPreprocessor
from .utils.metrics_helper import ClassificationMetrics, RegressionMetrics, MetricsRecommender
from .utils.model_selector import ModelSelector


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
        return Response({"metrics": calculated_metrics})

class RecommendMetric(APIView):
    def post(self, request):
        y = request.data.get("y")
        if not y:
            return Response({"error": "Target variable (y) must be provided."}, status=400)

        recommender = MetricsRecommender(y)
        recommendations = recommender.suggest_metric()
        return Response({"recommendations": recommendations})


class SelectModel(APIView):
    def post(self, request):
        X = request.data.get("X")
        y = request.data.get("y")
        task_type = request.data.get("task_type")  # classification or regression

        if not X or not y:
            return Response({"error": "X and y must be provided."}, status=400)

        selector = ModelSelector(X, y, task_type)
        best_model, best_score = selector.select_model()
        return Response({"best_model": str(best_model), "best_score": best_score})
