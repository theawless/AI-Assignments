"""
Handles regression techniques.
"""
import pathlib

import matplotlib.pyplot
import numpy
import sklearn.linear_model
import sklearn.metrics
import sklearn.neural_network
import sklearn.svm
import tqdm

import feature


def regressor_name(regressor):
    """
    Gets a human readable name for the regressor.
    :param regressor: regressor instance
    :return: name
    """
    name = regressor.__class__.__name__
    name = name.replace("Regressor", "")
    name = name.replace("Regression", "")
    return name


def regression(regressor, X_train, X_test, y_train):
    """
    Common regression function.
    :param regressor: regressor instance
    :param X_train: train features
    :param X_test: test features
    :param y_train: train output
    :return: predicted output
    """
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    return y_pred


def get_regressors():
    """
    List of regressors to be used.
    :return: regressors
    """
    return [
        sklearn.linear_model.HuberRegressor(),
        sklearn.svm.SVR(kernel="linear"),
        sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation="identity", solver="lbfgs"),
    ]


def plot_regression_for_multiple(data, n_test):
    """
    Plot the predictions for various regression techniques for multiple days in future.
    :param data: quandl data
    :param n_test: number of days for testing
    """
    data = feature.add_features(data)

    X, y = feature.build_sets(data, n_test)
    X_train, X_test, y_train, y_test = feature.divide_sets(X, y, n_test)
    for regressor in tqdm.tqdm(get_regressors()):
        y_pred_combined = []
        for day in range(1, n_test + 1):
            X, y = feature.build_sets(data, day)
            X_train, X_test, y_train, y_test = feature.divide_sets(X, y, n_test)
            X_train, X_test, _ = feature.select_features(X_train, X_test, y_train)

            y_pred = regression(regressor, X_train, X_test, y_train)
            y_pred_combined.append(y_pred[day - 1])
        r2_score = sklearn.metrics.r2_score(y_test, y_pred_combined)

        print(regressor_name(regressor), "r2_score", r2_score)
        matplotlib.pyplot.plot(y_pred_combined, label=regressor_name(regressor))
    matplotlib.pyplot.plot(y_test, label="actual")
    matplotlib.pyplot.title("Predictions of regressors for days in future")
    matplotlib.pyplot.xlabel("days")
    matplotlib.pyplot.ylabel("prices")
    matplotlib.pyplot.legend(loc="best")
    matplotlib.pyplot.savefig("plots/regression_for_multiple")
    matplotlib.pyplot.close()


def plot_regression_for_single(data, n_future, n_test):
    """
    Plot the predictions for various regression techniques for a single day in future.
    :param data: quandl data
    :param n_future: number of days in future to predict
    :param n_test: number of days for testing
    """
    data = feature.add_features(data)

    for day in tqdm.tqdm(range(1, n_future + 1)):
        X, y = feature.build_sets(data, day)
        X_train, X_test, y_train, y_test = feature.divide_sets(X, y, n_test)
        X_train, X_test, _ = feature.select_features(X_train, X_test, y_train)

        for regressor in get_regressors():
            y_pred = regression(regressor, X_train, X_test, y_train)
            r2_score = sklearn.metrics.r2_score(y_test, y_pred)

            print(regressor_name(regressor), "future day", day, "r2 score ", r2_score)
            matplotlib.pyplot.figure(day)
            matplotlib.pyplot.plot(y_pred, label=regressor_name(regressor))
        matplotlib.pyplot.plot(y_test, label="actual")
        matplotlib.pyplot.title("Predictions of regressors for day in future " + str(day))
        matplotlib.pyplot.xlabel("days")
        matplotlib.pyplot.ylabel("prices")
        matplotlib.pyplot.legend(loc="best")
        matplotlib.pyplot.savefig("plots/regression_for_single_" + str(day))
        matplotlib.pyplot.close()


if __name__ == "__main__":
    numpy.random.seed(0)
    pathlib.Path("data").mkdir(parents=True, exist_ok=True)
    pathlib.Path("plots").mkdir(parents=True, exist_ok=True)

    data = feature.get_quandl_data("WIKI", "GOOGL")
    plot_regression_for_single(data, 5, 100)
    plot_regression_for_multiple(data, 10)
