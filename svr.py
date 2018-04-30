import pandas as pd
from aggregate import get_dataset
from sklearn.svm import SVR

def run_model():
    x_train, x_test, y_train, y_test = get_dataset()

    clf = SVR()
    clf.fit(x_train, y_train)
    print clf.score(c_test, y_test)

if __name__ == "__main__":
    run_model()
