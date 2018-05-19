"""
Main script to demonstrate feature subset selection using various techniques on various datasets.
"""

import random

import numpy
from selection import exhaustive, sequential_forward, hill_climbing
from sklearn import datasets, linear_model
from sklearn.datasets import make_friedman1
from sklearn.metrics import r2_score


def select_features(X, indices):
    """
    Selects the features marked by the indices from numpy arrays.
    :param X: numpy array
    :param indices: list
    :return: numpy array with feature subset
    """
    X_subset = numpy.transpose(numpy.transpose(X)[[indices]])
    if len(X_subset.shape) == 1:
        X_subset = X_subset.reshape(-1, 1)
    return X_subset


def diabetes_features():
    """
    Optimise and display the features for a diabetes dataset using lasso regression
    and various feature subset selection methods.
    :return: None
    """
    diabetes = datasets.load_diabetes()
    X_train, X_test = diabetes.data[:-40], diabetes.data[-40:]
    y_train, y_test = diabetes.target[:-40], diabetes.target[-40:]

    def lasso_regression(feature_indices):
        """
        Fit lasso regression with train data and test its predictions.
        :param feature_indices: to select on the dataset
        :return: cost
        """
        model = linear_model.Lasso(alpha=0.6)
        model.fit(select_features(X_train, feature_indices), y_train)
        return model.score(select_features(X_test, feature_indices), y_test)

    hill_climbing_result = hill_climbing(X_train.shape[1], lasso_regression)
    sequential_forward_result = sequential_forward(X_train.shape[1], lasso_regression, 0.45)
    exhaustive_result = exhaustive(X_train.shape[1], lasso_regression)
    print("Diabetes dataset + hill climbing feature subset selection:\n", *hill_climbing_result)
    print("Diabetes dataset + sequential forward feature subset selection:\n", *sequential_forward_result)
    print("Diabetes dataset + exhaustive feature subset selection:\n", *exhaustive_result)


def friedman_features():
    """
    Optimise and display the features for a friedman dataset using linear regression
    and various feature subset selection methods.
    :return: None
    """
    data, target = make_friedman1(n_samples=500, n_features=10)
    X_train, X_test = data[:-200], data[-200:]
    y_train, y_test = target[:-200], target[-200:]

    def linear_regression(feature_indices):
        """
        Fit linear regression with train data and test its predictions.
        :param feature_indices: to select on the dataset
        :return: cost
        """
        model = linear_model.LinearRegression()
        model.fit(select_features(X_train, feature_indices), y_train)
        y_predicted = model.predict(select_features(X_test, feature_indices))
        # variance, 1 is the perfect score
        return r2_score(y_test, y_predicted)

    hill_climbing_result = hill_climbing(X_train.shape[1], linear_regression)
    sequential_forward_result = sequential_forward(X_train.shape[1], linear_regression, 0.65)
    exhaustive_result = exhaustive(X_train.shape[1], linear_regression)
    print("Friedman1 dataset + hill climbing feature subset selection:\n", *hill_climbing_result)
    print("Friedman1 dataset + sequential forward feature subset selection:\n", *sequential_forward_result)
    print("Friedman1 dataset + exhaustive feature subset selection:\n", *exhaustive_result)


if __name__ == "__main__":
    random.seed(0)
    diabetes_features()
    friedman_features()
