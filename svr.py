import pandas as pd
from aggregate import get_dataset
from sklearn.svm import SVR
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
def run_model():
    x_train, x_test, y_train, y_test = get_dataset()
    feat_select = SelectKBest(chi2, k=5000)
    X_new = feat_select.fit_transform(x_train, y_train)
    x_test_new = feat_select.transform(x_test)
    print X_new.shape
    print x_test_new.shape
    parameters = {'kernel':('linear','rbf'),'C':[1000,2000],'degree':[2,5],'coef0':[0,1]}
    regr = SVR(kernel = 'linear', C = 1000, degree = 5, coef0 = 0.5)
    # regr = GridSearchCV(svr,parameters)

    regr.fit(X_new, y_train)
    print regr.score(x_test_new, y_test)
    print regr.best_score
    print regr.best_params

if __name__ == "__main__":
    run_model()
