import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler


def rbf_kernel():
    X = np.c_[
        # Negative class
        (.3, -.8),
        (-1.5, -1),
        (-1.3, -.8),
        (-1.1, -1.3),
        (-1.2, -.3),
        (-1.3, -.5),
        (-.6, 1.1),
        (-1.4, 2.2),
        (1, 1),
        # Positive class
        (1.3, .8),
        (1.2, .5),
        (.2, -2),
        (.5, -2.4),
        (.2, -2.3),
        (0, -2.7),
        (1.3, 2.1)].T
    y = [-1] * 8 + [1] * 8
    gama_option = [1, 2, 4]
    print(X)

    plt.figure(1, figsize=(4*len(gama_option), 4))
    for i, gamma in enumerate(gama_option, 1):
        svm = SVC(kernel='rbf', gamma=gamma)
        svm.fit(X, y)
        plt.subplot(1, len(gama_option), i)
        plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired)
        plt.axis('tight')
        xx, yy = np.mgrid[-3:3:200j, -3:3:200j]
        z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)
        plt.contour(xx, yy, z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-.5, 0, .5])
        plt.title('gamma = %d' % gamma)

    plt.show()


def svm_regression():
    dataset = pd.read_csv('position_salaries.csv')
    dataset.head()
    print(dataset)

    # -->Feature selection and scaling
    # Store data as input and output
    x_orig = dataset.iloc[:, 1:2].values
    y_orig = dataset.iloc[:, 2:3].values

    # Scale features to normalize data
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    x = sc_x.fit_transform(x_orig)
    y = sc_y.fit_transform(y_orig)

    # Create SVM model and fit data
    regressor = SVR(kernel='rbf', C=1)
    regressor.fit(x, y.ravel())

    # Predict a new result
    prediction = np.array([[5.3]])
    y_pred = regressor.predict(sc_x.transform(prediction))
    y_pred = sc_y.inverse_transform(y_pred)
    print(y_pred)

    # Plot data and regression line
    y_pred = regressor.predict(x)
    y_pred = sc_y.inverse_transform(y_pred)
    plt.plot(x_orig, y_orig, 'bo')
    plt.plot(x_orig, y_pred)
    plt.show()


