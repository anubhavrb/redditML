import pandas as pd
from aggregate import read_dataset_from_file
from sklearn.ensemble import RandomForestRegressor

def run_model():
    x_train, x_test, y_train, y_test = read_dataset_from_file()

    regr = RandomForestRegressor(max_depth=2, random_state=0)
    regr.fit(x_train, y_train)
    print regr.score(x_test,y_test)


if __name__ == "__main__":
    run_model()
