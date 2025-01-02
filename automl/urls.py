from django.urls import path
from .views import UploadDataset, CalculateMetrics, RecommendMetric

urlpatterns = [
    path('upload/', UploadDataset.as_view(), name='upload'),
    path("metrics/calculate/", CalculateMetrics.as_view(), name="calculate_metrics"),
    path("metrics/recommend/", RecommendMetric.as_view(), name="recommend_metric"),
]
