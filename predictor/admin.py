from django.contrib import admin

from .models import WinePrediction


@admin.register(WinePrediction)
class WinePredictionAdmin(admin.ModelAdmin):
    list_display = ("user", "predicted_quality", "model_name", "created_at")
    list_filter = ("predicted_quality", "model_name")
    search_fields = ("user__username",)
    ordering = ("-created_at",)
