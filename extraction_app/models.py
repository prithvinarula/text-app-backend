"""_
Model for Project are created here
"""
from django.db import models


class File(models.Model):

    """
    File Class
    specifies the uploaded file storage location
    """

    file = models.FileField(upload_to="upload/")  # for creating file input
