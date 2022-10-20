from django.urls import path

from .api import views

urlpatterns = [
    path("", views.index, name="index"),
    path("blank/", views.blank_api, name="blank"),
    path("horcrux/", views.horcrux_api, name="horcrux"),
    path("cengage/", views.cengage_api, name="cengage"),
    path("ext_buisness/", views.bus_entity_api, name="ext_buisness"),
    path("ext_individual/", views.ind_entity_api, name="ext_individual"),
    path("ext_b_nlp/", views.bus_e_nlp_api, name="ext_b_nlp"),
    path("ext_i_nlp/", views.ind_e_nlp_api, name="ext_i_nlp"),
    path("consolidated_api", views.consolidated_api, name="consolidated_api"),
]
