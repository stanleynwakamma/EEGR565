# Stanley Nwakamma
# EEGR 565
# Assignment 2
# Introductory code on how to train ML module using KNN model.
# Google Drive link: https://drive.google.com/drive/folders/17-wzR8ZonH2wLCfuCqU3uTmF_Qv5cXC_?usp=sharing

import os
import numpy as np
import pandas as pd
from nltk.corpus import names
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import NMF


def assignment():
    # For CNN data csv file
    pd.set_option('display.max_columns', None)
    df = pd.read_csv('cnn_data_4_5.csv')
    print(df)

    # For pandemic text file
    df_pandemic = open("pandemic.txt", "r")
    print(df_pandemic.read())

    cv = CountVectorizer(stop_words="english", max_features=500)
    bag_of_words = cv.fit_transform(df.body)
    print(bag_of_words)

    cv_2 = CountVectorizer(df_pandemic)
    count_vector = cv.fit_transform(df_pandemic)

    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    print(words_freq)

    # To print all token and their frequencies
    for word, count in words_freq:
        print(word + ":", count)

    # To plot tokens and their frequencies
    words = []
    freqs = []
    for word, count in words_freq:
        words.append(word)
        freqs.append(count)
    plt.bar(np.arange(10), freqs[:10], align='center')
    plt.xticks(np.arange(10), words[:10])
    plt.ylabel('Frequency')
    plt.title("Top 10 Words")
    # plt.show()

    # To remove token with frequency less than 1000
    words_1 = []
    freqs_1 = []
    for word, count in words_freq:
        if count < 1000:
            words_1.append(word)
            freqs_1.append(count)
    plt.bar(np.arange(10), freqs_1[:10], align='center')
    plt.xticks(np.arange(10), words_1[:10])
    plt.ylabel('Frequency')
    plt.title("Top 10 Words")
    # plt.show()
