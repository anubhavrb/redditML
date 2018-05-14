import sys
import pandas as pd
from aggregate import get_dataset
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import r2_score

def run_model():
    x_train, x_test, y_train, y_test = get_dataset()

    clf = MLPClassifier()
    clf.fit(x_train, y_train)
    print clf.score(x_test, y_test)
    # predicted = regr.predict(x_test)
    #
    # print predicted.shape
    # print r2_score(y_test, predicted)


if __name__ == "__main__":
    run_model()
