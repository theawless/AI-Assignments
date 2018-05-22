"""
Handles the data and features.
"""
import pathlib

import matplotlib.pyplot
import numpy
import pandas
import quandl
import sklearn.feature_selection
import sklearn.linear_model
import sklearn.metrics
import sklearn.preprocessing
import stockstats


def select_features(X_train, X_test, y_train):
    """
    Feature subset selection using RFE and LinearRegression.
    :param X_train: train features
    :param X_test: test features
    :param y_train: train output
    :return: selected subsets of train features, test features
    """
    regressor = sklearn.linear_model.LinearRegression(n_jobs=-1)
    selector = sklearn.feature_selection.RFE(regressor)
    selector.fit(X_train, y_train)
    return X_train[:, selector.support_], X_test[:, selector.support_], selector.support_


def divide_sets(X, y, n_test):
    """
    Divide into the training and test sets.
    :param X: features
    :param y: output
    :param n_test: number of days for testing
    :return: train features, test features, train output, test output
    """
    X_train = X[:-n_test]
    X_test = X[-n_test:]
    y_train = y[:-n_test]
    y_test = y[-n_test:]

    scaler = sklearn.preprocessing.RobustScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    scaler.fit(y_train)
    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_test)
    return X_train, X_test, y_train, y_test


def build_sets(data, n_future):
    """
    Divides the data into features and output
    :param data: quandl data
    :param n_future: number of days in future to predict
    :return: features, output
    """
    output = pandas.DataFrame()
    open_column = "open"
    prediction_column = "prediction"
    output[prediction_column] = data[open_column].shift(-n_future)
    output = output[:-n_future]
    features = data[:-n_future]
    return features.values, output.values


def add_features(data):
    """
    Add features and build output.
    :param data: quandl data
    :return: stockstats data with additional features
    """
    data = stockstats.StockDataFrame.retype(data)
    original_columns = list(data.columns.values)
    artificial_columns = ["macd", "middle", "boll", "kdjk", "open_2_sma", "rsi_6", "wr_6", "cci", "adxr", "vr"]
    for column in artificial_columns:
        data.get(column)
    data = data[original_columns + artificial_columns]
    data = data.replace([numpy.inf, -numpy.inf], numpy.nan)
    data = data.dropna()
    return data


def get_quandl_data(market, company):
    """
    Gets lots of data from quandl.
    :param market: stock market id
    :param company: company id
    :return: data
    """
    filename = "data/" + company + ".csv"

    try:
        # load if already exists
        return pandas.read_csv(filename, parse_dates=True, index_col=0)
    except Exception:
        original_columns = ["open", "close", "high", "low", "volume"]
        data = quandl.get(market + '/' + company)
        data.columns = data.columns.str.lower()
        data = data[original_columns]
        data.to_csv(filename)
        return data


def plot_feature_selection(data, n_future, n_test):
    """
    Plots the effect of feature selection on prediction.
    :param data: quandl data
    :param n_future: number of days in future to predict
    :param n_test: number of days for testing
    """
    data = add_features(data)
    X, y = build_sets(data, n_future)
    X_train, X_test, y_train, y_test = divide_sets(X, y, n_test)
    X_train_fs, X_test_fs, selected = select_features(X_train, X_test, y_train)

    regressor = sklearn.linear_model.LinearRegression(n_jobs=-1)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    r2_score = sklearn.metrics.r2_score(y_test, y_pred)
    regressor.fit(X_train_fs, y_train)
    y_pred_fs = regressor.predict(X_test_fs)
    r2_score_fs = sklearn.metrics.r2_score(y_test, y_pred_fs)

    print("all")
    print("features", data.columns.values)
    print("r2_score", r2_score)
    print("selected")
    print("features", data.columns.values[selected])
    print("r2_score", r2_score_fs)
    matplotlib.pyplot.plot(y_pred, label="all")
    matplotlib.pyplot.plot(y_pred_fs, label="selected")
    matplotlib.pyplot.plot(y_test, label="actual")
    matplotlib.pyplot.title("Effect of feature selection")
    matplotlib.pyplot.xlabel("days")
    matplotlib.pyplot.ylabel("prices")
    matplotlib.pyplot.legend(loc="best")
    matplotlib.pyplot.savefig("plots/feature_selection")
    matplotlib.pyplot.close()


if __name__ == "__main__":
    numpy.random.seed(0)
    pathlib.Path("data").mkdir(parents=True, exist_ok=True)
    pathlib.Path("plots").mkdir(parents=True, exist_ok=True)

    data = get_quandl_data("WIKI", "GOOGL")
    plot_feature_selection(data, 3, 100)
