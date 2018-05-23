"""
Handles recurrent neural networks.
"""
import pathlib

import keras.layers.core
import keras.layers.recurrent
import keras.models
import matplotlib.pyplot
import numpy
import sklearn.metrics
import tqdm

import feature


def recurrent_name(recurrent):
    """
    Gets a human readable name for the recurrent.
    :param recurrent: recurent instance
    :return: name
    """
    return ' '.join(recurrent.__name__.split('_'))


def recurrent_stacked_lstm(X_train, X_test, y_train):
    """
    Recurrent stacked LSTM network.
    :param X_train: train features
    :param X_test: test features
    :param y_train: train output
    :return: predicted output
    """
    network = keras.Sequential()
    network.add(keras.layers.LSTM(50, input_shape=X_train.shape[1:], return_sequences=True, dropout=0.3))
    network.add(keras.layers.LSTM(100, dropout=0.2))
    network.add(keras.layers.core.Dense(1))
    network.compile(loss="mae", optimizer="rmsprop", metrics=["mse"])

    network.fit(X_train, y_train, batch_size=768, epochs=10, validation_split=0.1)
    y_pred = network.predict(X_test)
    return y_pred


def recurrent_lstm(X_train, X_test, y_train):
    """
    Recurrent LSTM network.
    :param X_train: train features
    :param X_test: test features
    :param y_train: train output
    :return: predicted output
    """
    network = keras.Sequential()
    network.add(keras.layers.LSTM(150, input_shape=X_train.shape[1:], dropout=0.2))
    network.add(keras.layers.core.Dense(1))
    network.compile(loss="mae", optimizer="rmsprop", metrics=["mse"])

    network.fit(X_train, y_train, batch_size=768, epochs=10, validation_split=0.1)
    y_pred = network.predict(X_test)
    return y_pred


def get_recurrent_predictors():
    """
    List of recurrents to be used.
    :return: recurrent functions
    """
    return [
        recurrent_lstm,
        recurrent_stacked_lstm,
    ]


def build_sequence(X, y, size):
    """
    Builds a windowed sequence.
    :param X: features
    :param y: output
    :param size: window size
    :return: feature sequence, output sequence
    """
    Xs = []
    for i in range(X.shape[0] - size - 1):
        Xs.append(X[i:(i + size)])
    Xs = numpy.array(Xs)
    y = y[size + 1:]
    return Xs, y


def plot_recurrent_for_multiple(data, n_test, n_past):
    """
    Plot the predictions for various recurrent techniques for multiple days in future.
    :param data: quandl data
    :param n_test: number of days for testing
    :param n_past: number of days to remember in context
    """
    X, y = feature.build_sets(data, n_test)
    X_train, X_test, y_train, y_test = feature.divide_sets(X, y, n_test + n_past + 1)
    X_test, y_test = build_sequence(X_test, y_test, n_past)
    for recurrent in get_recurrent_predictors():
        y_pred_combined = []
        for day in tqdm.tqdm(range(1, n_test + 1)):
            X, y = feature.build_sets(data, day)
            X_train, X_test, y_train, y_test = feature.divide_sets(X, y, n_test + n_past + 1)
            X_train, y_train = build_sequence(X_train, y_train, n_past)
            X_test, y_test = build_sequence(X_test, y_test, n_past)

            y_pred = recurrent(X_train, X_test, y_train)
            y_pred_combined.append(y_pred[day - 1])
        r2_score = sklearn.metrics.r2_score(y_test, y_pred_combined)

        print(recurrent_name(recurrent), "r2_score", r2_score)
        matplotlib.pyplot.plot(y_pred_combined, label=recurrent_name(recurrent))
    matplotlib.pyplot.plot(y_test, label="actual")
    matplotlib.pyplot.title("Predictions of recurrent for days in future")
    matplotlib.pyplot.xlabel("days")
    matplotlib.pyplot.ylabel("prices")
    matplotlib.pyplot.legend(loc="best")
    matplotlib.pyplot.savefig("plots/recurrent_for_multiple")
    matplotlib.pyplot.close()


def plot_recurrent_for_single(data, n_future, n_test, n_past):
    """
    Plot the predictions for various recurrent techniques for a single day in future.
    :param data: quandl data
    :param n_future: number of days in future to predict
    :param n_test: number of days for testing
    :param n_past: number of days to remember in context
    """
    for day in tqdm.tqdm(range(1, n_future + 1)):
        X, y = feature.build_sets(data, day)
        X_train, X_test, y_train, y_test = feature.divide_sets(X, y, n_test + n_past + 1)
        X_train, y_train = build_sequence(X_train, y_train, n_past)
        X_test, y_test = build_sequence(X_test, y_test, n_past)

        for recurrent in get_recurrent_predictors():
            y_pred = recurrent(X_train, X_test, y_train)
            r2_score = sklearn.metrics.r2_score(y_test, y_pred)

            print(recurrent_name(recurrent), "future day", day, "r2 score ", r2_score)
            matplotlib.pyplot.figure(day)
            matplotlib.pyplot.plot(y_pred, label=recurrent_name(recurrent))
        matplotlib.pyplot.plot(y_test, label="actual")
        matplotlib.pyplot.title("Predictions of recurrent for day in future " + str(day))
        matplotlib.pyplot.xlabel("days")
        matplotlib.pyplot.ylabel("prices")
        matplotlib.pyplot.legend(loc="best")
        matplotlib.pyplot.savefig("plots/recurrent_for_single_" + str(day))
        matplotlib.pyplot.close()


if __name__ == "__main__":
    numpy.random.seed(0)
    pathlib.Path("data").mkdir(parents=True, exist_ok=True)
    pathlib.Path("plots").mkdir(parents=True, exist_ok=True)

    data = feature.get_quandl_data("WIKI", "GOOGL")
    plot_recurrent_for_single(data, 5, 100, 500)
    plot_recurrent_for_multiple(data, 10, 500)
