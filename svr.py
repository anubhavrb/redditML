import pandas as pd
from aggregate import get_dataset
from sklearn.svm import SVR
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
def run_model():
    x_train, x_test, y_train, y_test = get_dataset()


    regr = SVR(kernel = 'linear')

    # regr = GridSearchCV(svr,parameters)

    regr.fit(x_train, y_train)
    print regr.score(x_test_new, y_test)
    print regr.best_score
    print regr.best_params

if __name__ == "__main__":
    run_model()
