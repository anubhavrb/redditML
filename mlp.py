import pandas as pd
from aggregate import read_dataset_from_file
from sklearn.neural_network import MLPRegressor

def run_model():
    x_train, x_test, y_train, y_test = read_dataset_from_file()
    print x_train.shape
    regr = MLPRegressor()
    regr.fit(x_train, y_train)
    print regr.score(x_test,y_test)


if __name__ == "__main__":
    run_model()
