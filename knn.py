import pandas as pd
from aggregate import get_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
def run_model():
    x_train, x_test, y_train, y_test = get_dataset()


    clf = KNeighborsClassifier()
    # regr = GridSearchCV(svr,parameters)

    clf.fit(x_train, y_train)
    print clf.score(x_test, y_test)



if __name__ == "__main__":
    run_model()
