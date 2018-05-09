import pandas as pd
from aggregate import get_dataset
from sklearn.neighbors import KNeighborsRegressor
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
    regr = KNeighborsRegressor()
    # regr = GridSearchCV(svr,parameters)
    print x_train.shape
    regr.fit(x_train, y_train)
    print regr.score(x_test, y_test)



if __name__ == "__main__":
    run_model()
