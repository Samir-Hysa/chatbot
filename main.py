import nltk
import tflearn
import tensorflow as tf
import random
import numpy as np
import json
from nltk.stem.lancaster  import LancasterStemmer
stemmer = LancasterStemmer()

words = []
labels = []
docs_x = []
docs_y = []

with open("data/intents.json") as file:
    data = json.load(file)

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(pattern)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])



