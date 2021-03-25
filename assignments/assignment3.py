# Stanley Nwakamma
# EEGR 565
# Assignment 3
# Building a spam classifier by two methods: unsupervised learning (K-Means Clustering) and multinomial Naïve Bayes.
# Google Drive link: https://drive.google.com/drive/folders/17-wzR8ZonH2wLCfuCqU3uTmF_Qv5cXC_?usp=sharing
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.corpus import names
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt


def assignment():
    # Load data
    pd.set_option('display.max_columns', None)
    df = pd.read_csv("spam.csv")
    print(df.head())

    # Function to return True if the string is an alphabetic string
    def letters_only(astr):
        return astr.isalpha()

    # Function to perform word lemmatizer
    def cleaned_text(email):
        all_names = set(x.lower() for x in names.words())
        lemmatizer = WordNetLemmatizer()
        cleaned = []
        for messages in email:
            cleaned.append(' '.join(lemmatizer.lemmatize(word.lower()) for word in messages.split()
                                    if letters_only(word) and word.lower() not in all_names))
        return cleaned

    # List to store label (ham & spam) as 1 & 0
    target = []
    for labels in df.label:
        if labels == 'ham':
            target.append(1)
        else:
            target.append(0)

    # Returns a list of messages after it has been lemmatized
    cleaned_emails = cleaned_text(df.message)

    # Split dataset into training set and test set
    X_train, X_test, Y_train, Y_test = train_test_split(cleaned_emails, target, test_size=0.3, random_state=42)

    """
    Un-supervised Learning: Using K-Means clustering to make a prediction
    """
    # Creating a TF-IDF feature vector for clustering
    tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words="english", max_features=500)
    term_docs_train = tfidf_vectorizer.fit_transform(X_train)
    term_docs_test = tfidf_vectorizer.transform(X_test)

    # Predicting using K-Means Clustering
    Kmean = KMeans(n_clusters=2).fit(term_docs_train)
    all_predictions = Kmean.predict(term_docs_test, Y_test)

    category_0 = term_docs_test[all_predictions == 0].A
    category_1 = term_docs_test[all_predictions == 1].A
    category_0_max = category_0.argmax(axis=0)
    category_1_max = category_1.argmax(axis=0)
    category_0_pairs = [(token, category_0[category_0_max[idx], idx]) for token, idx in
                        tfidf_vectorizer.vocabulary_.items()]
    category_1_pairs = [(token, category_1[category_1_max[idx], idx]) for token, idx in
                        tfidf_vectorizer.vocabulary_.items()]

    category_0_pairs = sorted(category_0_pairs, key=lambda x: x[1], reverse=True)
    category_1_pairs = sorted(category_1_pairs, key=lambda x: x[1], reverse=True)

    print("\nThe first 25 spam tokens are: ")
    for count in range(25):
        print(category_0_pairs[count])

    print("\n\nThe first 25 ham tokens are: ")
    for count in range(25):
        print(category_1_pairs[count])

    """
    Supervised Learning: Using multinomial Naïve Bayes to make a prediction
    """
    clf = MultinomialNB(alpha=1, fit_prior=True)
    clf.fit(term_docs_train, Y_train)

    # Calculate the accuracy of classifier based on test data
    accuracy = clf.score(term_docs_test, Y_test)
    print("\nAccuracy using Naive bayes: ", accuracy)

