from django.contrib import messages
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404, redirect, render

from . import ml_model
from .forms import RegisterForm, WinePredictionForm
from .models import WinePrediction


def is_admin(user):
    return user.is_staff or user.is_superuser


def register_view(request):
    """Registration page for new (regular) users."""
    if request.user.is_authenticated:
        return redirect("predict")

    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            auth_login(request, user)
            messages.success(request, f"Welcome, {user.username}! Your account has been created.")
            return redirect("predict")
    else:
        form = RegisterForm()

    return render(request, "predictor/register.html", {"form": form})


def login_view(request):
    """Login page for authenticated access."""
    if request.user.is_authenticated:
        return redirect("predict")

    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            auth_login(request, user)
            messages.success(request, f"Logged in as {user.username}.")
            next_url = request.POST.get("next") or request.GET.get("next")
            if is_admin(user):
                return redirect(next_url or "admin_dashboard")
            return redirect(next_url or "predict")
        messages.error(request, "Invalid username or password.")
    else:
        form = AuthenticationForm()

    return render(request, "predictor/login.html", {"form": form})


@login_required
def logout_view(request):
    auth_logout(request)
    messages.info(request, "You have been logged out.")
    return redirect("login")


@login_required
def predict_view(request):
    """Prediction form containing the 11 wine measurements."""
    if request.method == "POST":
        form = WinePredictionForm(request.POST)
        if form.is_valid():
            prediction = form.save(commit=False)
            prediction.user = request.user

            feature_dict = {
                "fixed acidity": form.cleaned_data["fixed_acidity"],
                "volatile acidity": form.cleaned_data["volatile_acidity"],
                "citric acid": form.cleaned_data["citric_acid"],
                "residual sugar": form.cleaned_data["residual_sugar"],
                "chlorides": form.cleaned_data["chlorides"],
                "free sulfur dioxide": form.cleaned_data["free_sulfur_dioxide"],
                "total sulfur dioxide": form.cleaned_data["total_sulfur_dioxide"],
                "density": form.cleaned_data["density"],
                "pH": form.cleaned_data["ph"],
                "sulphates": form.cleaned_data["sulphates"],
                "alcohol": form.cleaned_data["alcohol"],
            }

            try:
                quality, confidence = ml_model.predict_quality_with_confidence(feature_dict)
            except FileNotFoundError as exc:
                messages.error(request, str(exc))
                return render(request, "predictor/predict.html", {"form": form})

            metadata = ml_model.get_metadata()
            prediction.predicted_quality = quality
            prediction.model_name = metadata.get("model_name", "Unknown")
            prediction.save()

            request.session["last_confidence"] = confidence
            return redirect("result", pk=prediction.pk)
    else:
        form = WinePredictionForm()

    return render(request, "predictor/predict.html", {"form": form})


@login_required
def result_view(request, pk):
    """Result page showing the predicted quality score."""
    prediction = get_object_or_404(WinePrediction, pk=pk)
    if prediction.user != request.user and not is_admin(request.user):
        messages.error(request, "You do not have permission to view that prediction.")
        return redirect("predict")

    confidence = request.session.pop("last_confidence", None)
    return render(
        request,
        "predictor/result.html",
        {"prediction": prediction, "confidence": confidence},
    )


@login_required
def user_dashboard_view(request):
    """User dashboard showing the user's previous predictions."""
    predictions = WinePrediction.objects.filter(user=request.user)
    return render(request, "predictor/user_dashboard.html", {"predictions": predictions})


@user_passes_test(is_admin, login_url="login")
def admin_dashboard_view(request):
    """Administrator dashboard showing registered users and prediction records."""
    users = User.objects.all().order_by("-date_joined")
    predictions = WinePrediction.objects.select_related("user").all()[:200]
    return render(
        request,
        "predictor/admin_dashboard.html",
        {
            "users": users,
            "predictions": predictions,
            "total_users": users.count(),
            "total_predictions": WinePrediction.objects.count(),
        },
    )
