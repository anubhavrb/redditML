import sys
import pandas as pd
from aggregate import get_dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

def run_model_regressor():
    print "Random Forest Regressor"
    x_train, x_test, y_train, y_test = get_dataset()

    regr = RandomForestRegressor()
    regr.fit(x_train, y_train)
    print regr.score(x_test,y_test)

def run_model_classifier():
    print "Random Forest Classifier"
    x_train, x_test, y_train, y_test = get_dataset()

    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    print clf.score(x_test,y_test)


if __name__ == "__main__":
    if sys.argv[1] == 'R'.lower():
        run_model_regressor()
    else:
        run_model_classifier()
