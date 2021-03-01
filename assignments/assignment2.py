# Stanley Nwakamma
# EEGR 565
# Assignment 1
# Introductory code on how to train ML module using KNN model.
# Google Drive link: https://drive.google.com/drive/folders/17-wzR8ZonH2wLCfuCqU3uTmF_Qv5cXC_?usp=sharing

import numpy as np
from nltk.corpus import names
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import NMF


def assignment():
    groups = fetch_20newsgroups()
