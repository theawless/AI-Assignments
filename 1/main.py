import numpy
from sklearn import datasets, linear_model
from sklearn.datasets import make_friedman1
from sklearn.metrics import r2_score

from selection import exhaustive, sequential_forward


def select_features(X, indices):
    # for numpy arrays only
    X_subset = numpy.transpose(numpy.transpose(X)[[indices]])
    if len(X_subset.shape) == 1:
        X_subset = X_subset.reshape(-1, 1)
    return X_subset


def diabetes_features():
    diabetes = datasets.load_diabetes()
    X_train, X_test = diabetes.data[:-40], diabetes.data[-40:]
    y_train, y_test = diabetes.target[:-40], diabetes.target[-40:]

    def lasso_regression(feature_indices):
        model = linear_model.Lasso(alpha=0.6)
        model.fit(select_features(X_train, feature_indices), y_train)
        return model.score(select_features(X_test, feature_indices), y_test)

    exhaustive_subset = exhaustive(X_train.shape[1], lasso_regression)
    sequential_forward_subset = sequential_forward(X_train.shape[1], lasso_regression, 0.45)
    print("Diabetes dataset + exhaustive feature subset selection:\n", exhaustive_subset)
    print("Diabetes dataset + sequential forward feature subset selection:\n", sequential_forward_subset)


def friedman_features():
    data, target = make_friedman1(n_samples=500, n_features=10)
    X_train, X_test = data[:-200], data[-200:]
    y_train, y_test = target[:-200], target[-200:]

    def linear_regression(feature_indices):
        model = linear_model.LinearRegression()
        model.fit(select_features(X_train, feature_indices), y_train)
        Y_predicted = model.predict(select_features(X_test, feature_indices))
        # variance, 1 is the perfect score
        return r2_score(y_test, Y_predicted)

    exhaustive_subset = exhaustive(X_train.shape[1], linear_regression)
    sequential_forward_subset = sequential_forward(X_train.shape[1], linear_regression, 0.65)
    print("Friedman1 dataset + exhaustive feature subset selection:\n", exhaustive_subset)
    print("Friedman1 dataset + sequential forward feature subset selection:\n", sequential_forward_subset)


if __name__ == "__main__":
    diabetes_features()
    friedman_features()
