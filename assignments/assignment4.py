# Stanley Nwakamma
# EEGR 565
# Assignment 4
# Performing a Gaussian Naïve Bayes Classifier
# Google Drive link: https://drive.google.com/drive/folders/17-wzR8ZonH2wLCfuCqU3uTmF_Qv5cXC_?usp=sharing

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def assignment():
    # Load and read dataset
    dataset = pd.read_csv("abalone.data")
    print(dataset)

    # Get the first column which is the sex column
    sex_dataset = dataset.iloc[:, 0]

    # Get the data in a dataset
    data = dataset.iloc[:, 1:7]

    # Get the target data
    target = dataset.iloc[:, 8]

    # creating a label encoder object to encode the sex
    le = preprocessing.LabelEncoder()
    sex_encoded = le.fit_transform(sex_dataset)

    # Reshaping the data to include in dataset
    sex_encoded = sex_encoded.reshape(len(sex_encoded), 1)

    # Combining the encoded data into the dataset
    data_stack = np.hstack((data, sex_encoded))

    # Separating the data into a training and test set
    X_train, X_test, y_train, y_test = train_test_split(data_stack, target, test_size=0.3, random_state=42)

    # create a GaussianNB classifier
    gnb = GaussianNB()

    # Train the model using the training sets
    gnb.fit(X_train, y_train)
    accuracy = gnb.score(X_test, y_test)
    print("Accuracy before restructuring target label: ", accuracy)

    """Restructuring the target labels and retraining my model"""
    new_target = [1 if 1 <= age <= 4 else 2 if 5 <= age <= 15 else 3 for age in target]
    X_train, X_test, y_train, y_test = train_test_split(data_stack, new_target, test_size=0.3, random_state=42)

    # create a GaussianNB classifier
    gnb = GaussianNB()

    # Train the model using the training sets
    gnb.fit(X_train, y_train)
    accuracy = gnb.score(X_test, y_test)
    print("Accuracy after restructuring target label: ", accuracy)
