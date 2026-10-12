from django.conf import settings
from django.db import models


class WinePrediction(models.Model):
    """A single wine-quality prediction submitted by an authenticated user."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="predictions",
    )

    # 11 physicochemical features (Id is excluded from modelling, per the report)
    fixed_acidity = models.FloatField()
    volatile_acidity = models.FloatField()
    citric_acid = models.FloatField()
    residual_sugar = models.FloatField()
    chlorides = models.FloatField()
    free_sulfur_dioxide = models.FloatField()
    total_sulfur_dioxide = models.FloatField()
    density = models.FloatField()
    ph = models.FloatField(verbose_name="pH")
    sulphates = models.FloatField()
    alcohol = models.FloatField()

    predicted_quality = models.IntegerField()
    model_name = models.CharField(max_length=100, default="Random Forest")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.user.username} — quality {self.predicted_quality} ({self.created_at:%Y-%m-%d %H:%M})"

    def as_feature_dict(self):
        """Return the 11 features in the exact order the model expects."""
        return {
            "fixed acidity": self.fixed_acidity,
            "volatile acidity": self.volatile_acidity,
            "citric acid": self.citric_acid,
            "residual sugar": self.residual_sugar,
            "chlorides": self.chlorides,
            "free sulfur dioxide": self.free_sulfur_dioxide,
            "total sulfur dioxide": self.total_sulfur_dioxide,
            "density": self.density,
            "pH": self.ph,
            "sulphates": self.sulphates,
            "alcohol": self.alcohol,
        }
