"""
Handles recurrent neural networks.
"""

import keras.layers.core
import keras.layers.recurrent
import keras.models
import matplotlib.pyplot
import numpy
import sklearn.metrics
import tqdm

import feature


def build_sequence(X, y, size):
    Xs = []
    for i in range(X.shape[0] - size - 1):
        Xs.append(X[i:(i + size)])
    return numpy.array(Xs), numpy.array(y[size + 1:])


def recurrent_lstm(X_train, X_test, y_train, y_test):
    """
    Recurrent LSTM network.
    :param X_train: train features
    :param X_test: test features
    :param y_train: train output
    :param y_test: test output
    :return: predicted output and score
    """
    network = keras.Sequential()
    network.add(keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=(X_train.shape[1:])))
    network.add(keras.layers.core.Dense(1))
    network.compile(loss="mse", optimizer="rmsprop")

    network.fit(X_train, y_train, batch_size=512, epochs=10, validation_split=0.05)
    y_pred = network.predict(X_test)
    r2_score = sklearn.metrics.r2_score(y_test, y_pred)
    return y_pred, r2_score


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

        y_pred, r2_score = recurrent_lstm(X_train, X_test, y_train, y_test)

        print("recurrent lstm", "future day", day, "r2 score ", r2_score)
        matplotlib.pyplot.figure(day)
        matplotlib.pyplot.plot(y_pred, label="recurrent lstm")
        matplotlib.pyplot.plot(y_test, label="actual")
        matplotlib.pyplot.title("Predictions of recurrent for day in future " + str(day))
        matplotlib.pyplot.xlabel('days')
        matplotlib.pyplot.ylabel('prices')
        matplotlib.pyplot.legend(loc="best")
    matplotlib.pyplot.show()


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

    y_pred_combined = []
    for day in tqdm.tqdm(range(1, n_test + 1)):
        X, y = feature.build_sets(data, day)
        X_train, X_test, y_train, y_test = feature.divide_sets(X, y, n_test + n_past + 1)
        X_train, y_train = build_sequence(X_train, y_train, n_past)
        X_test, y_test = build_sequence(X_test, y_test, n_past)

        y_pred, r2_score = recurrent_lstm(X_train, X_test, y_train, y_test)
        y_pred_combined.append(y_pred[day - 1])
    r2_score = sklearn.metrics.r2_score(y_test, y_pred_combined)

    print("recurrent lstm", "r2_score", r2_score)
    matplotlib.pyplot.plot(y_pred_combined, label="recurrent lstm")
    matplotlib.pyplot.plot(y_test, label="actual")
    matplotlib.pyplot.title("Predictions of recurrent for days in future")
    matplotlib.pyplot.xlabel('days')
    matplotlib.pyplot.ylabel('prices')
    matplotlib.pyplot.legend(loc="best")
    matplotlib.pyplot.show()


if __name__ == "__main__":
    numpy.random.seed(0)

    data = feature.get_quandl_data("WIKI", "GOOGL")
    plot_recurrent_for_single(data, 5, 100, 50)
    plot_recurrent_for_multiple(data, 10, 50)
