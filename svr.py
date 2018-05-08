import pandas as pd
from aggregate import read_dataset_from_file
from sklearn.svm import SVR

def run_model():
    x_train, x_test, y_train, y_test = read_dataset_from_file()

    clf = SVR()
    clf.fit(x_train, y_train)
    print clf.score(x_test, y_test)

if __name__ == "__main__":
    run_model()
