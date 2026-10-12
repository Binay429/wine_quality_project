from django.contrib.auth.models import User
from rest_framework import serializers

from .models import WinePrediction


class WinePredictionSerializer(serializers.ModelSerializer):
    """
    Serializer for the wine-quality prediction API.

    Accepts the 11 physicochemical measurements on input; ``predicted_quality``,
    ``model_name``, ``user`` and ``created_at`` are set by the view/model and
    returned read-only in the response.
    """

    username = serializers.CharField(source="user.username", read_only=True)

    class Meta:
        model = WinePrediction
        fields = [
            "id",
            "username",
            "fixed_acidity",
            "volatile_acidity",
            "citric_acid",
            "residual_sugar",
            "chlorides",
            "free_sulfur_dioxide",
            "total_sulfur_dioxide",
            "density",
            "ph",
            "sulphates",
            "alcohol",
            "predicted_quality",
            "model_name",
            "created_at",
        ]
        read_only_fields = ["id", "username", "predicted_quality", "model_name", "created_at"]


class UserSummarySerializer(serializers.ModelSerializer):
    """Lightweight user representation for the admin API endpoint."""

    prediction_count = serializers.IntegerField(source="predictions.count", read_only=True)

    class Meta:
        model = User
        fields = ["id", "username", "email", "is_staff", "is_superuser", "date_joined", "prediction_count"]
