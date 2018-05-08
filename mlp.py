import pandas as pd
from aggregate import get_dataset
from sklearn.neural_network import MLPRegressor

def run_model():
    x_train, x_test, y_train, y_test = get_dataset()
    regr = MLPRegressor()
    regr.fit(x_train, y_train)
    print regr.score(x_test,y_test)


if __name__ == "__main__":
    run_model()
