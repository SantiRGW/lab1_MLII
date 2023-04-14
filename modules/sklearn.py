from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
#Reduction

def load_nmist_dataset():
    mnist = fetch_openml('mnist_784')
    X, y = mnist["data"], mnist["target"]

    X_numbers_0_8 = X[(y == '0') | (y == '8')]
    y_numbers_0_8 = y[(y == '0') | (y == '8')]
    X_numbers_0_8 = X_numbers_0_8.to_numpy()
    X_numbers_0_8 = X_numbers_0_8.reshape(X_numbers_0_8.shape[0], -1)

    split = int(X_numbers_0_8.shape[0] * 0.8)
    X_train, y_train = X_numbers_0_8[:split], y_numbers_0_8[:split]
    X_test, y_test = X_numbers_0_8[split:], y_numbers_0_8[split:]
    return X_train, y_train, X_test, y_test

def apply_logistic_regression():
    X_train, y_train, X_test, y_test = load_nmist_dataset()
    model_LR = LogisticRegression(random_state=0).fit(X_train, y_train)
    return model_LR, model_LR.score(X_test, y_test)

def apply_logistic_regression_any(X_train, y_train, X_test, y_test):
    model_LR = LogisticRegression(random_state=0).fit(X_train, y_train)
    return model_LR, model_LR.score(X_test, y_test)


