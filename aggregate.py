import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer as TV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

"""
Master function to return the vectorized training and test sets.
"""
def get_dataset():
    df = pd.read_csv("2016_2017.csv")
    df = df.fillna("")
    df = df.sample(frac=0.25)
    x_train, x_test, y_train, y_test = split_data(df)

    #print x_train.shape, y_train.shape
    #print x_test.shape, y_test.shape

    v_train, v_test = vectorize_text(x_train['title'], x_test['title'])
    x_train = x_train[['day_of_year', 'day_of_week', 'hour', 'minute']]
    x_test = x_test[['day_of_year', 'day_of_week', 'hour', 'minute']]

    #print "FINISHED VECTORIZING"
    #print x_test
    #print x_train.shape,v_train.shape
    #x_train = pd.concat([x_train, v_train], ignore_index=True, axis=1)
    x_train = pd.DataFrame(np.hstack([x_train, v_train]))
    x_test = pd.DataFrame(np.hstack([x_test, v_test]))
    #x_test = pd.concat([x_test, v_test], ignore_index=True, axis=1)

    #print x_train.shape, y_train.shape
    #print x_test.shape, y_test.shape
    #print x_test
    return x_train, x_test, y_train, y_test

"""
Function that splits df into train and test sets based on a 80:20 split.
"""
def split_data(df):
    y = df['score']
    x = df.drop('score', axis = 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test

"""
Function that vectorizes the text data in the dataset.
"""
def vectorize_text(x_train, x_test):
    vect = TV(strip_accents='unicode', analyzer='word', max_features=1000)
    text_train = vect.fit_transform(x_train).toarray()
    text_test = vect.transform(x_test).toarray()
    text_train, text_test = pd.DataFrame(text_train), pd.DataFrame(text_test)
    return text_train, text_test


# def random_forest(X_train,y_train, X_test, y_test):
#     regr = RandomForestRegressor(max_depth=2, random_state=0)
#     regr.fit(X_train, y_train)
#     predicted = regr.predict(X_test)
#     return r2_score(y_test, predicted)
#
#
# def main():
#     # df = parse_csv()
#     # df = df.dropna(how = 'any')
#     # df = df.sample(frac=1)
#     # df = df.reset_index(drop = True)
#     # df.to_pickle('aggregate.pkl')
#     df = pd.read_pickle('aggregate.pkl')
#     print df.shape
#     train_set = df.iloc[0:int(df.shape[0]*0.8)]
#     test_set = df.iloc[int(df.shape[0]*0.8):df.shape[0]-1]
#     x_train = train_set.drop(columns = ['score','created_utc'])
#     x_test = test_set.drop(columns = ['score','created_utc'])
#     text_train, text_test,count_vect,tfidf_transformer = vectorize_text(x_train,x_test)
#     print text_train.shape
#     print random_forest(text_train,train_set['score'],text_test,test_set['score'])


if __name__ == '__main__':
    get_dataset()
