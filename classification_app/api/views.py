import json
import pickle
import numpy as np
import pandas as pd
import tensorflow_hub as hub
from json2html import *

# Create your views here.
from django.shortcuts import render

# from .mlmodel import getTop3
from dotenv import load_dotenv
import os
import spacy
from ..mlmodel import cengage, horcrux, bus_entity_model, ind_entity_model, bus_e_nlp, ind_e_nlp, blank,tfidf_models

from django.http import HttpResponse, JsonResponse

# import spacy.cli

# spacy.cli.download("en_core_web_lg")


def index(request):
    return render(request, "base.html")


def blank_api(request):
    if request.method == "POST":
        to_be_classified = request.POST["to_be_classified"]
        to_be_classified = [to_be_classified]
        # blank_response = blank(to_be_classified)
        # result.append({"blank": blank_response})
        infoFromJson = blank(to_be_classified)
        result = json2html.convert(json=infoFromJson)
        return render(request, "base.html", {"result": result})


def horcrux_api(request):
    if request.method == "POST":
        to_be_classified = request.POST["to_be_classified"]
        to_be_classified = [to_be_classified]
        infoFromJson = horcrux(to_be_classified)
        result = json2html.convert(json=infoFromJson)
        return render(request, "base.html", {"result": result})


def cengage_api(request):
    if request.method == "POST":
        to_be_classified = request.POST["to_be_classified"]
        to_be_classified = [to_be_classified]
        infoFromJson = cengage(to_be_classified)
        result = json2html.convert(json=infoFromJson)
        return render(request, "base.html", {"result": result})


def bus_entity_api(request):
    if request.method == "POST":
        to_be_classified = request.POST["to_be_classified"]
        to_be_classified = [to_be_classified]
        infoFromJson = bus_entity_model(to_be_classified)
        result = json2html.convert(json=infoFromJson)
        return render(request, "base.html", {"result": result})


def ind_entity_api(request):
    if request.method == "POST":
        to_be_classified = request.POST["to_be_classified"]
        to_be_classified = [to_be_classified]
        infoFromJson = ind_entity_model(to_be_classified)
        result = json2html.convert(json=infoFromJson)
        return render(request, "base.html", {"result": result})


def bus_e_nlp_api(request):
    if request.method == "POST":
        to_be_classified = request.POST["to_be_classified"]
        to_be_classified = [to_be_classified]
        infoFromJson = bus_e_nlp(to_be_classified)
        result = json2html.convert(json=infoFromJson)
        return render(request, "base.html", {"result": result})


def ind_e_nlp_api(request):
    if request.method == "POST":
        to_be_classified = request.POST["to_be_classified"]
        to_be_classified = [to_be_classified]
        infoFromJson = ind_e_nlp(to_be_classified)
        result = json2html.convert(json=infoFromJson)
        return render(request, "base.html", {"result": result})


def tfidf_model_api(request):
    if request.method == "POST":
        to_be_classified = request.POST["to_be_classified"]
        to_be_classified = [to_be_classified]
        infoFromJson = tfidf_models(to_be_classified)
        result = json2html.convert(json=infoFromJson)
        return render(request, "base.html", {"result": result})   

def json_transformer(input, model, output_arr):
    print("Transformer")
    print(input)
    print(model)
    for text in input:
        print(text)
        print(input[text])
        classifications = input[text]
        for classification in classifications:
            output = {}
            output["model"] = model
            output["input"] = text
            output["classification"] = classification
            output["probability"] = classifications[classification]
            output["component"] = "entity"
            print(classification)
            print(classifications[classification])
            output_arr.append(output)
    print(output_arr)
    return output_arr


# {
#             "input": "Laptop",
#             "model": "BLANK",
#             "probability": "98.2%",
#             "classification": "Assets",
#             "riskReviewTime": "2022-03-31 17:11:53", "component": "entity"
#           }


def consolidated_api(request):
    print(request)
    # print(request.POST)
    print(request.GET)
    result = []
    if request.method == "POST":
        to_be_classified = request.POST["to_be_classified"]
        to_be_classified = [to_be_classified]
    elif request.method == "GET":
        to_be_classified = request.GET["data"]
        to_be_classified = [to_be_classified]

    print(to_be_classified)
    blank_response = blank(to_be_classified)
    result = json_transformer(blank_response, "BLANK", result)

    cengage_response = cengage(to_be_classified)
    result = json_transformer(cengage_response, "CENGAGE", result)

    horcrux_response = horcrux(to_be_classified)
    result = json_transformer(horcrux_response, "HORCRUX", result)

    bert_bus_response = bus_entity_model(to_be_classified)
    result = json_transformer(bert_bus_response, "BERT_BUS", result)

    bert_ind_response = ind_entity_model(to_be_classified)
    result = json_transformer(bert_ind_response, "BERT_IND", result)

    nlp_bus_response = bus_e_nlp(to_be_classified)
    result = json_transformer(nlp_bus_response, "NLP_BUS", result)

    nlp_ind_response = ind_e_nlp(to_be_classified)
    result = json_transformer(nlp_ind_response, "NLP_IND", result)

    tfidf_model_response = tfidf_models(to_be_classified)
    result = json_transformer(tfidf_model_response, "tfidf_model", result)

    return JsonResponse({"data": result}, safe=False)
