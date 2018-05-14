import pandas as pd
from aggregate import get_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
def run_model():
    x_train, x_test, y_train, y_test = get_dataset()
    feat_select = SelectKBest(chi2, k=10000)
    X_new = feat_select.fit_transform(x_train, y_train)
    x_test_new = feat_select.transform(x_test)

    regr = KNeighborsRegressor()
    regr.fit(x_train, y_train)
    print "kNN Regressor with 10000 Feature selection"
    print regr.score(x_test, y_test)

if __name__ == "__main__":
    run_model()
