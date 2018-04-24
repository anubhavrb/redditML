import numpy as np
import pandas as pd
import datetime as dt
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.feature_extraction.text import TfidfTransformer as TF
from sklearn.feature_extraction.text import TfidfVectorizer as TV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
def parse_csv():
    df_2016 = pd.read_csv("2016.csv")
    df_2017 = pd.read_csv("2017.csv")
    df = pd.concat([df_2016,df_2017])
    return df

def vectorize_text(x_train, x_test):

    count_vect = CV(strip_accents='unicode', analyzer = 'word', max_features = 100000)
    X_train_counts = count_vect.fit_transform(x_train["title"])
    tfidf_transformer = TF()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    X_new_counts = count_vect.transform(x_test['title'])
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    return X_train_tfidf, X_new_tfidf,count_vect,tfidf_transformer


"""
Function that adds day, hour, and minute columns to the df.
"""
def add_time_columns(row):
    utc_time = row['created_utc']
    day = dt.datetime.fromtimestamp(int(utc_time)).strftime('%A')
    hour = dt.datetime.fromtimestamp(int(utc_time)).strftime('%-H')
    minute = dt.datetime.fromtimestamp(int(utc_time)).strftime('%-M')
    return pd.Series([day, hour, minute], index = ['day', 'hour', 'minute'])
def random_forest(X_train,y_train, X_test, y_test):
    regr = RandomForestRegressor(max_depth=2, random_state=0)
    regr.fit(X_train, y_train)
    predicted = regr.predict(X_test)
    return r2_score(y_test, predicted)


def main():
    # df = parse_csv()
    # df = df.dropna(how = 'any')
    # df = df.sample(frac=1)
    # df = df.reset_index(drop = True)
    # df.to_pickle('aggregate.pkl')
    df = pd.read_pickle('aggregate.pkl')
    print df.shape
    train_set = df.iloc[0:int(df.shape[0]*0.8)]
    test_set = df.iloc[int(df.shape[0]*0.8):df.shape[0]-1]
    x_train = train_set.drop(columns = ['score','created_utc'])
    x_test = test_set.drop(columns = ['score','created_utc'])
    text_train, text_test,count_vect,tfidf_transformer = vectorize_text(x_train,x_test)
    print text_train.shape
    print random_forest(text_train,train_set['score'],text_test,test_set['score'])


if __name__ == '__main__':
    main()
