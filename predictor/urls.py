from django.urls import path

from . import views

urlpatterns = [
    path("", views.predict_view, name="home"),
    path("register/", views.register_view, name="register"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("predict/", views.predict_view, name="predict"),
    path("result/<int:pk>/", views.result_view, name="result"),
    path("dashboard/", views.user_dashboard_view, name="user_dashboard"),
    path("admin-dashboard/", views.admin_dashboard_view, name="admin_dashboard"),
]
