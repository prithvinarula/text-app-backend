"""
Url File containing the urls and the view that they refer to.
"""
from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("image", views.ima, name="ima"),
    path("file_upload", views.file_upload, name=""),
]
