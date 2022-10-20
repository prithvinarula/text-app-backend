from cmath import inf
from dotenv import load_dotenv
import os
from sklearn.linear_model import LogisticRegression
import re

import gensim
import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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

load_dotenv()

path_to_test_file = os.getenv("path_to_test_file")
path_to_file = os.chdir(path_to_test_file)

df = pd.read_csv(os.path.join(path_to_test_file, "data.csv"))
df = df[pd.notnull(df["text"])]
df = df[pd.notnull(df["tag"])]
df = df.reset_index()

REPLACE_BY_SPACE_RE = re.compile(r"[/(){}\[\]\|@,;]")
BAD_SYMBOLS_RE = re.compile("[^0-9a-z #+_]")
STOPWORDS = set(stopwords.words("english"))

load_dotenv()


model = hub.load(os.path.join(path_to_test_file, "universal-sentence-encoder_4"))
# model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4?tf-hub-format=compressed")

sample_data = pd.read_excel(os.path.join(path_to_test_file, "testdata1.xlsx"), sheet_name="Sheet1")
Individual_Entity = pd.read_excel(
    os.path.join(path_to_test_file, "Scrubbing Data Classification_v0.2.xlsx"), sheet_name="Individual Entity"
)
Business_Entity = pd.read_excel(
    os.path.join(path_to_test_file, "Scrubbing Data Classification_v0.2.xlsx"), sheet_name="Business Entity"
)


# trainDF = sample_data

# LineItems = trainDF["LineItem"].tolist()

Business_Entity_cls = Business_Entity["Classifications"].tolist()
Individual_Entity_cls = Individual_Entity["Classifications"].tolist()
nlp = spacy.load("en_core_web_lg")

# Business_Entity_embeddings = model(Business_Entity_cls)
# Individual_Entity_embeddings = model(Individual_Entity_cls)
# line_embeddings = model(LineItems)


def clean_text(text):
    """
    text: a string

    return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text  # HTML decoding
    text = text.lower()  # lowercase text
    # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = REPLACE_BY_SPACE_RE.sub(" ", text)
    # delete symbols which are in BAD_SYMBOLS_RE from text
    text = BAD_SYMBOLS_RE.sub("", text)
    # delete stopwors from text
    text = " ".join(word for word in text.split() if word not in STOPWORDS)
    return text


df["text"] = df["text"].apply(clean_text)

wv = gensim.models.KeyedVectors.load_word2vec_format(
    os.path.join(path_to_test_file, "GoogleNews-vectors-negative300.bin.gz"),
    binary=True,
)
wv.init_sims(replace=True)


s = set(wv.index_to_key)


def word_averaging(wv, words):
    """_summary_

    Args:
        wv (_type_): _description_
        words (_type_): _description_

    Returns:
        _type_: _description_
    """
    all_words, mean = set(), []
    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in s:
            mean.append(wv.vectors[wv.key_to_index[word]])
            all_words.add(wv.key_to_index[word])

    if not mean:
        return np.zeros(
            wv.vector_size,
        )

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean


def word_averaging_list(wv, text_list):
    """_summary_

    Args:
        wv (_type_): _description_
        text_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.vstack([word_averaging(wv, text) for text in text_list])


def w2v_tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text, language="english"):
        for word in nltk.word_tokenize(sent, language="english"):
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens


train, test = train_test_split(df, test_size=0.3, random_state=42)
test_tokenized = test.apply(lambda r: w2v_tokenize_text(r["text"]), axis=1).values
train_tokenized = train.apply(lambda r: w2v_tokenize_text(r["text"]), axis=1).values
X_train_word_average = word_averaging_list(wv, train_tokenized)
X_test_word_average = word_averaging_list(wv, test_tokenized)


logreg = LogisticRegression(n_jobs=1, C=1e5, max_iter=1000000)
logreg = logreg.fit(X_train_word_average, train["tag"])
y_pred = logreg.predict(X_test_word_average)
score = accuracy_score(y_pred, test["tag"])


def getTop3(text):
    """_summary_

    Args:
        text (_type_): _description_

    Returns:
        _type_: _description_
    """
    processedInput = word_averaging(wv, w2v_tokenize_text(text))
    prob = logreg.predict_proba([processedInput])
    prob = prob[0]
    labels = logreg.classes_
    res = {}
    for i, p in enumerate(prob):
        res[p] = labels[i]
    prob.sort()
    ans = []
    keys = prob[-3:]
    for i in keys:
        ans.append([res[i], i])
    return ans


def blank(to_be_classified):
    for line in to_be_classified:
        l_x = [i.strip() for i in line.replace('"', "").split(",")]

    d = {}
    for word in l_x:
        key = word
        y_pred = getTop3(word)
        d_1 = {}
        li = []
    for list in y_pred:
        key_1 = list[0]
        value = list[1]
        li.append(value)
        d_1[value] = key_1

        li.sort(reverse=True)

    for item in li:
        key_2 = d_1[item]  # 99
        del d_1[item]
        item = "{:.2%}".format(item)
        d_1[key_2] = item
        d[key] = d_1
        s1 = json.dumps(d)
        infoFromJson = json.loads(s1)

    return infoFromJson


def horcrux(to_be_classified):
    path1 = os.path.join(path_to_test_file, "modelf.pkl")
    path2 = os.path.join(path_to_test_file, "dataframe.pkl")
    with open(path1, "rb") as file:
        model = pickle.load(file)
    with open(path2, "rb") as file:
        dataf = pickle.load(file)

    for line in to_be_classified:
        l_x = [i.strip() for i in line.replace('"', "").split(",")]
    d = {}
    for word in l_x:
        key = word
        to_be_classified = [word]
        prediction = model.predict(to_be_classified)
        prediction = dataf[prediction[0]]
        value = prediction
        d[key] = {value: "100%"}
    s1 = json.dumps(d)
    infoFromJson = json.loads(s1)
    # result = json2html.convert(json=infoFromJson)
    return infoFromJson
    # return render(request, "base.html", {"result": result})  # return JsonResponse(d, safe=False)


def cengage(to_be_classified):
    pathm = os.path.join(path_to_test_file, "model.pkl")
    pathd = os.path.join(path_to_test_file, "dict.pkl")
    with open(pathm, "rb") as f:
        modelc = pickle.load(f)
    with open(pathd, "rb") as f:
        datafc = pickle.load(f)

        for line in to_be_classified:
            l_x = [i.strip() for i in line.replace('"', "").split(",")]

        d = {}
        for word in l_x:
            key = word
            to_classified = [word]
            predictionc = modelc.predict(to_classified)
            predictionc = datafc[predictionc[0]]
            value = predictionc
            d[key] = {value: "100%"}
            s1 = json.dumps(d)
            infoFromJson = json.loads(s1)
        return infoFromJson


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def bus_entity_model(to_be_classified):
    for line in to_be_classified:
        l_x = [i.strip() for i in line.replace('"', "").split(",")]
        result = {}
        for line in l_x:
            query_vec = model([line])[0]
            coreclass = []
            Similarity = []
            for class_itm in Business_Entity_cls:
                sim = cosine(query_vec, model([class_itm])[0])
                coreclass.append(class_itm)
                Similarity.append(sim)
            dict1 = {"Classlabel": coreclass, "MatchProbability": Similarity}
            df = pd.DataFrame(dict1)
            final_df = df.sort_values(by=["MatchProbability"], ascending=False)
            dict_df = final_df.head(3).to_dict()
            lis1 = []
            lis2 = []
            for obj1 in dict_df["Classlabel"].values():
                lis1.append(obj1)
            for obj2 in dict_df["MatchProbability"].values():
                lis2.append(obj2)
                final_lis = [f"{i*100:.2f}%" for i in lis2]
            dictionary = {k: v for k, v in zip(lis1, final_lis)}
            expected_output = {line: dictionary}
            d1 = []
            d2 = []
            for keyss in expected_output.keys():
                d1.append(keyss)
            for valuess in expected_output.values():
                d2.append(valuess)
            new_dict = {kk: vv for kk, vv in zip(d1, d2)}  # dict(zip(d1, d2))
            result.update(new_dict)
            s1 = json.dumps(result)
            infoFromJson = json.loads(s1)
        return infoFromJson
        # return render(request, "base.html", {"result": result}) # return JsonResponse(result, safe=False)


def ind_entity_model(to_be_classified):
    for line in to_be_classified:
        l_x = [i.strip() for i in line.replace('"', "").split(",")]
        result = {}
        for line in l_x:
            query_vec = model([line])[0]
            coreclass = []
            Similarity = []
            for class_itm in Individual_Entity_cls:
                sim = cosine(query_vec, model([class_itm])[0])
                coreclass.append(class_itm)
                Similarity.append(sim)
            dict1 = {"Classlabel": coreclass, "MatchProbability": Similarity}
            df = pd.DataFrame(dict1)
            final_df = df.sort_values(by=["MatchProbability"], ascending=False)
            dict_df = final_df.head(3).to_dict()
            lis1 = []
            lis2 = []
            for obj1 in dict_df["Classlabel"].values():
                lis1.append(obj1)
            for obj2 in dict_df["MatchProbability"].values():
                lis2.append(obj2)
            final_lis = [f"{i*100:.2f}%" for i in lis2]
            dictionary = {k: v for k, v in zip(lis1, final_lis)}
            expected_output = {line: dictionary}
            d1 = []
            d2 = []
            for keyss in expected_output.keys():
                d1.append(keyss)
            for valuess in expected_output.values():
                d2.append(valuess)
            new_dict = {kk: vv for kk, vv in zip(d1, d2)}  # dict(zip(d1, d2))
            result.update(new_dict)
            s1 = json.dumps(result)
            infoFromJson = json.loads(s1)
        return infoFromJson


def bus_e_nlp(to_be_classified):
    for line in to_be_classified:
        l_x = [i.strip() for i in line.replace('"', "").split(",")]
        result = {}
        for line in l_x:
            line_token = nlp(line)
            coreclass = []
            Similarity = []
            for class_itm in Business_Entity_cls:
                class_itm_token = nlp(class_itm)
                sim = line_token.similarity(class_itm_token)
                coreclass.append(class_itm)
                Similarity.append(sim)
            dict1 = {"Classlabel": coreclass, "MatchProbability": Similarity}
            df = pd.DataFrame(dict1)
            final_df = df.sort_values(by=["MatchProbability"], ascending=False)
            dict_df = final_df.head(3).to_dict()
            lis1 = []
            lis2 = []
            for obj1 in dict_df["Classlabel"].values():
                lis1.append(obj1)
            for obj2 in dict_df["MatchProbability"].values():
                lis2.append(obj2)
                final_lis = [f"{i*100:.2f}%" for i in lis2]
            dictionary = {k: v for k, v in zip(lis1, final_lis)}
            expected_output = {line: dictionary}
            d1 = []
            d2 = []
            for keyss in expected_output.keys():
                d1.append(keyss)
            for valuess in expected_output.values():
                d2.append(valuess)
            new_dict = {kk: vv for kk, vv in zip(d1, d2)}
            result.update(new_dict)
            s1 = json.dumps(result)
            infoFromJson = json.loads(s1)
        return infoFromJson


def ind_e_nlp(to_be_classified):
    for line in to_be_classified:
        l_x = [i.strip() for i in line.replace('"', "").split(",")]
        result = {}
        for line in l_x:
            line_token = nlp(line)
            coreclass = []
            Similarity = []
            for class_itm in Individual_Entity_cls:
                class_itm_token = nlp(class_itm)
                sim = line_token.similarity(class_itm_token)
                coreclass.append(class_itm)
                Similarity.append(sim)
            dict1 = {"Classlabel": coreclass, "MatchProbability": Similarity}
            df = pd.DataFrame(dict1)
            final_df = df.sort_values(by=["MatchProbability"], ascending=False)
            dict_df = final_df.head(3).to_dict()
            lis1 = []
            lis2 = []
            for obj1 in dict_df["Classlabel"].values():
                lis1.append(obj1)
            for obj2 in dict_df["MatchProbability"].values():
                lis2.append(obj2)
            final_lis = [f"{i*100:.2f}%" for i in lis2]
            dictionary = {k: v for k, v in zip(lis1, final_lis)}
            expected_output = {line: dictionary}
            d1 = []
            d2 = []
            for keyss in expected_output.keys():
                d1.append(keyss)
            for valuess in expected_output.values():
                d2.append(valuess)
            new_dicts = {kk: vv for kk, vv in zip(d1, d2)}
            print(type(new_dicts))
            result.update(new_dicts)
            s1 = json.dumps(result)
            infoFromJson = json.loads(s1)
        return infoFromJson
