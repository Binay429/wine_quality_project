"""
REST API for the Wine Quality Prediction system, built with Django REST Framework.

Endpoints:
    POST /api/predict/              submit measurements, get a prediction back
    GET  /api/predictions/          the authenticated user's own prediction history
    GET  /api/predictions/<id>/     a single prediction (owner or admin only)
    GET  /api/admin/predictions/    all predictions (admin/staff only)
    GET  /api/admin/users/          all registered users (admin/staff only)

All endpoints require an authenticated session (log in via /login/ first, then
browse /api/ in the same browser) or HTTP Basic Auth (e.g. from Postman/curl),
per the DEFAULT_AUTHENTICATION_CLASSES in settings.py.
"""

from django.contrib.auth.models import User
from rest_framework import generics, permissions
from rest_framework.exceptions import PermissionDenied
from rest_framework.response import Response

from . import ml_model
from .models import WinePrediction
from .serializers import UserSummarySerializer, WinePredictionSerializer


class IsStaffOrSuperuser(permissions.BasePermission):
    """Only staff/superuser accounts may access admin API endpoints."""

    def has_permission(self, request, view):
        return bool(request.user and (request.user.is_staff or request.user.is_superuser))


class PredictAPIView(generics.CreateAPIView):
    """
    Wine quality prediction endpoint.

    Send the 11 wine measurements with POST (use the form below) and the
    response contains the predicted quality score and the model used.
    """

    serializer_class = WinePredictionSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, *args, **kwargs):
        """Usage guide shown when the endpoint is opened in a browser."""
        return Response({
            "message": "Send a POST request with the 11 wine measurements to get a predicted quality score.",
            "required_fields": [
                "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide",
                "density", "ph", "sulphates", "alcohol",
            ],
            "example_request": {
                "fixed_acidity": 7.4, "volatile_acidity": 0.7, "citric_acid": 0.0,
                "residual_sugar": 1.9, "chlorides": 0.076, "free_sulfur_dioxide": 11.0,
                "total_sulfur_dioxide": 34.0, "density": 0.9978, "ph": 3.51,
                "sulphates": 0.56, "alcohol": 9.4,
            },
            "response_fields": ["predicted_quality", "model_name", "created_at"],
        })

    def perform_create(self, serializer):
        feature_dict = {
            "fixed acidity": serializer.validated_data["fixed_acidity"],
            "volatile acidity": serializer.validated_data["volatile_acidity"],
            "citric acid": serializer.validated_data["citric_acid"],
            "residual sugar": serializer.validated_data["residual_sugar"],
            "chlorides": serializer.validated_data["chlorides"],
            "free sulfur dioxide": serializer.validated_data["free_sulfur_dioxide"],
            "total sulfur dioxide": serializer.validated_data["total_sulfur_dioxide"],
            "density": serializer.validated_data["density"],
            "pH": serializer.validated_data["ph"],
            "sulphates": serializer.validated_data["sulphates"],
            "alcohol": serializer.validated_data["alcohol"],
        }
        quality, _ = ml_model.predict_quality_with_confidence(feature_dict)
        metadata = ml_model.get_metadata()
        serializer.save(
            user=self.request.user,
            predicted_quality=quality,
            model_name=metadata.get("model_name", "Unknown"),
        )


class MyPredictionListAPIView(generics.ListAPIView):
    """The authenticated user's own prediction history."""

    serializer_class = WinePredictionSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return WinePrediction.objects.filter(user=self.request.user)


class PredictionDetailAPIView(generics.RetrieveAPIView):
    """A single prediction — the owner or an admin/staff user may view it."""

    serializer_class = WinePredictionSerializer
    permission_classes = [permissions.IsAuthenticated]
    queryset = WinePrediction.objects.all()

    def get_object(self):
        obj = super().get_object()
        if obj.user != self.request.user and not (self.request.user.is_staff or self.request.user.is_superuser):
            raise PermissionDenied("You do not have permission to view this prediction.")
        return obj


class AdminPredictionListAPIView(generics.ListAPIView):
    """All prediction records — admin/staff only."""

    serializer_class = WinePredictionSerializer
    permission_classes = [permissions.IsAuthenticated, IsStaffOrSuperuser]
    queryset = WinePrediction.objects.all()


class AdminUserListAPIView(generics.ListAPIView):
    """All registered users — admin/staff only."""

    serializer_class = UserSummarySerializer
    permission_classes = [permissions.IsAuthenticated, IsStaffOrSuperuser]
    queryset = User.objects.all().order_by("-date_joined")
