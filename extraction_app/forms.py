"""
Baic Upload Form made in Django
"""
from django.forms import ModelForm

from .models import File


class FileData(ModelForm):
    """_summary_

    Args:
        ModelForm (_type_): _description_
    """

    class Meta:
        """_summary_"""

        model = File

        fields = "__all__"
