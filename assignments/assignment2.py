# Stanley Nwakamma
# EEGR 565
# Assignment 2
# Newsgroup Topic modelling with Natural Language Processing
# Google Drive link: https://drive.google.com/drive/folders/17-wzR8ZonH2wLCfuCqU3uTmF_Qv5cXC_?usp=sharing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def assignment():
    # For CNN data csv file
    pd.set_option('display.max_columns', None)
    df = pd.read_csv('cnn_data_4_5.csv')
    # print(df)
    cv = CountVectorizer(stop_words="english", max_features=500)
    bag_of_words = cv.fit_transform(df.body)
    # print(bag_of_words)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    # print(words_freq)

    # For pandemic text file
    df_pandemic = pd.read_csv("pandemic.txt")
    cv_2 = CountVectorizer(stop_words="english", max_features=500)
    bag_of_words2 = cv_2.fit_transform(df_pandemic.Terms)
    sum_words2 = bag_of_words2.sum(axis=0)
    words_freq2 = [(word2, sum_words2[0, idx]) for word2, idx in cv_2.vocabulary_.items()]
    words_freq2 = sorted(words_freq2, key=lambda x: x[1], reverse=True)
    # print(words_freq2)

    # To print all token and their frequencies for the CNN file
    # for word, count in words_freq:
    # print(word + ":", count)

    # To plot tokens and their frequencies from CNN file
    words = []
    freqs = []
    for word, count in words_freq:
        words.append(word)
        freqs.append(count)
    plt.bar(np.arange(10), freqs[:10], align='center')
    plt.xticks(np.arange(10), words[:10])
    plt.ylabel('Frequency')
    plt.title("Top 10 Words")
    plt.show()

    # To remove token with frequency more than 1000 from CNN file
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
    plt.show()

    # To plot token with relation to the pandemic file
    words_2 = []
    freqs_2 = []
    for word, count in words_freq:
        for word_2, count_2 in words_freq2:
            if word == word_2:
                words_2.append(word)
                freqs_2.append(count)
    plt.bar(np.arange(10), freqs_2[:10], align='center')
    plt.xticks(np.arange(10), words_2[:10])
    plt.ylabel('Frequency')
    plt.title("Top 10 Words")
    plt.show()
