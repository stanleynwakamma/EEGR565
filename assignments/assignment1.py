# Stanley Nwakamma
# EEGR 565
# Assignment 1
# Introductory code on how to train ML module using KNN model.
# Google Drive link: https://drive.google.com/drive/folders/17-wzR8ZonH2wLCfuCqU3uTmF_Qv5cXC_?usp=sharing

from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def assignment():
    # Load dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = read_csv(url, names=names)

    # shape
    print(dataset.shape)
    # head
    print(dataset.head(20))

    # Split-out validation dataset
    array = dataset.values
    X = array[:, 0:4]
    y = array[:, 4]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

    # Train (Cross validation)
    model = KNeighborsClassifier()
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    print(cv_results.mean())

    # Make predictions on validation dataset
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)

    # Evaluate predictions
    print(accuracy_score(Y_validation, predictions))
