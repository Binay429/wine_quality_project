from django.urls import path

from . import api_views

urlpatterns = [
    path("predict/", api_views.PredictAPIView.as_view(), name="api_predict"),
    path("predictions/", api_views.MyPredictionListAPIView.as_view(), name="api_my_predictions"),
    path("predictions/<int:pk>/", api_views.PredictionDetailAPIView.as_view(), name="api_prediction_detail"),
    path("admin/predictions/", api_views.AdminPredictionListAPIView.as_view(), name="api_admin_predictions"),
    path("admin/users/", api_views.AdminUserListAPIView.as_view(), name="api_admin_users"),
]
