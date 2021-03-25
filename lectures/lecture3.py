import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing, metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def lecture():
    # Load data
    df = pd.read_csv("play_weather_dataset.csv")
    df.head()

    # creating labelEncoder
    le = preprocessing.LabelEncoder()

    # Converting string labels into numbers.
    weather_encoded = le.fit_transform(df['weather'])
    temp_encoded = le.fit_transform(df['temp'])
    label_encoded = le.fit_transform(df['play'])
    print('Weather:', weather_encoded)
    print("Temp:", temp_encoded)
    print("Play:", label_encoded)

    # Combining weather and temp into single list of tuples
    features = list(zip(weather_encoded, temp_encoded))
    print(features)

    # Create a naive bayes Gaussian Classifier
    model = GaussianNB()

    # Train the model using the training sets
    model.fit(features, label_encoded)

    # Predict Output
    predicted = model.predict([[0, 2]])  # 0:Overcast, 2:Mild
    print("Predicted Value:", predicted)


def lecture_2():
    # Load dataset
    wine = datasets.load_wine()

    # print the names of the 13 features
    print("Features: ", wine.feature_names)

    # print the label type of wine(class_0, class_1, class_2)
    print("Labels: ", wine.target_names)

    # Show first few rows of data
    print(wine.data.shape)
    print(wine.data[0:5])

    # Show targets
    print(wine.target)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=109)

    # Create a Gaussian Classifier
    gnb = GaussianNB()

    # Train the model using the training sets
    gnb.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = gnb.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(y_pred)