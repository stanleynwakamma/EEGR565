import numpy as np
from nltk.corpus import names
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import NMF


def lecture():
    groups = fetch_20newsgroups()
    # print(groups.keys())
    # print(groups['target_names'])
    # print('Here is the group target:', groups.target)
    # print(np.unique(groups.target))
    # print(groups.data[0])
    # print(groups.target[0])
    # print(groups.target_names[groups.target[0]])
    cv = CountVectorizer(stop_words="english", max_features=500)
    bag_of_words = cv.fit_transform(groups.data)
    print(bag_of_words)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    print(words_freq)
    for word, count in words_freq:
        print(word + ":", count)

    words = []
    freqs = []
    for word, count in words_freq:
        words.append(word)
        freqs.append(count)

    # Plot frequency
    plt.bar(np.arange(10), freqs[:10], align='center')
    plt.xticks(np.arange(10), words[:10])
    plt.ylabel('Frequency')
    plt.title("Top 10 Words")
    plt.show()

    # Test if token is a word
    def letters_only(astr):
        return astr.isalpha()

    # Remove names from words and perform word lemmatization
    cleaned = []
    all_names = set(x.lower() for x in names.words())
    lemmatizer = WordNetLemmatizer()
    for post in groups.data[:250]:
        cleaned.extend(list(lemmatizer.lemmatize(word.lower()) for word in post.split()
                            if letters_only(word) and word.lower() not in all_names))
    cleaned_bag_of_words = cv.fit_transform(cleaned)
    print(cv.get_feature_names())
    transformed = cv.fit_transform(cleaned)
    nmf = NMF(n_components=100, random_state=43).fit(transformed)

    for topic_idx, topic in enumerate(nmf.components_):
        label = '{}: '.format(topic_idx)
        print(label, " ".join([cv.get_feature_names()[i] for i in topic.argsort()[:-9:-1]]))
