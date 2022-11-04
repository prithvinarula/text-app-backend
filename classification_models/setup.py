from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import ensemble

import pandas, xgboost, numpy, textblob, string  # $ pip install textblob
import csv,re,nltk
import time
import pandas as pd

# ---Install TF-Hub.
# !pip3 install --upgrade tensorflow-gpu
# !pip3 install tensorflow-hub

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import random

import os
os.chdir(r"C:\Users\PNarula\Documents\STGI")
os.environ["TFHUB_CACHE_DIR"] = r"C:\Users\PNarula\Documents\STGI"

import tensorflow_hub as hub
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4?tf-hub-format=compressed")
test = model(["The quick brown fox jumps over the lazy dog.","I am a sentence for which I would like to get its embedding"])

print(test)


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))