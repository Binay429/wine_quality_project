from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

from .models import WinePrediction


class RegisterForm(UserCreationForm):
    """Registration form for new (regular) users."""

    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ["username", "email", "password1", "password2"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs.setdefault("class", "form-control")

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data["email"]
        if commit:
            user.save()
        return user


class WinePredictionForm(forms.ModelForm):
    """Form containing the 11 wine physicochemical measurements."""

    class Meta:
        model = WinePrediction
        fields = [
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
        ]
        labels = {
            "fixed_acidity": "Fixed acidity",
            "volatile_acidity": "Volatile acidity",
            "citric_acid": "Citric acid",
            "residual_sugar": "Residual sugar",
            "chlorides": "Chlorides",
            "free_sulfur_dioxide": "Free sulfur dioxide",
            "total_sulfur_dioxide": "Total sulfur dioxide",
            "density": "Density",
            "ph": "pH",
            "sulphates": "Sulphates",
            "alcohol": "Alcohol (%)",
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, field in self.fields.items():
            field.widget.attrs.setdefault("class", "form-control")
            field.widget.attrs.setdefault("step", "any")
            field.required = True

    def clean(self):
        """Validate that every measurement is numeric and non-negative
        (pH and density are allowed their natural ranges)."""
        cleaned_data = super().clean()
        non_negative_fields = [
            "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
            "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide",
            "sulphates", "alcohol",
        ]
        for field_name in non_negative_fields:
            value = cleaned_data.get(field_name)
            if value is not None and value < 0:
                self.add_error(field_name, "Value cannot be negative.")
        return cleaned_data
