import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer as TV
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer
import operator
import nltk
from nltk.corpus import wordnet
import sys
reload(sys)
sys.setdefaultencoding("utf-8")


"""
Save the sampled data to pickle file
"""
def save_sampled():
    df = pd.read_csv("2016_2017.csv")
    df = df.fillna("")
    df = df.sample(frac=0.01)
<<<<<<< HEAD
    df = df.reset_index()
    median = df['score'].median()
    mean = df['score'].mean()
    df['above_mean'] = df.apply(lambda row: 0 if row['score'] < mean else 1, axis=1)
    df['above_median'] = df.apply(lambda row: 0 if row['score'] < median else 1, axis=1)
    print df
=======

>>>>>>> f8cd2ac4c27183be16bf1e7d83b6bd581c317307
    df.to_pickle('ten_percent_sampled.pkl')
    print ("done.")
    return 0
"""
Return Vectorized training and test sets from file
"""
def save_dataset_from_file():
    df = pd.read_pickle('ten_percent_sampled.pkl')
    x_train, x_test, y_train, y_test = split_data(df)

    v_train, v_test = vectorize_text(x_train['title'], x_test['title'])
    x_train = x_train[['day_of_year', 'day_of_week', 'hour', 'minute']]
    x_test = x_test[['day_of_year', 'day_of_week', 'hour', 'minute']]

    x_train = pd.DataFrame(np.hstack([x_train, v_train]))
    x_test = pd.DataFrame(np.hstack([x_test, v_test]))

    x_train.to_pickle('ten_percent_sampled_x_train.pkl')
    x_test.to_pickle('ten_percent_sampled_x_test.pkl')
    y_train.to_pickle('ten_percent_sampled_y_train.pkl')
    y_test.to_pickle('ten_percent_sampled_y_test.pkl')
    return x_train, x_test, y_train, y_test


def read_dataset_from_file():
    df = pd.read_pickle('ten_percent_sampled.pkl')

    x_train, x_test, y_train, y_test = split_data(df)
    print (x_train)
    v_train, v_test = vectorize_text(x_train['title'], x_test['title'])
    x_train = x_train[['day_of_year', 'day_of_week', 'hour', 'minute']]
    x_test = x_test[['day_of_year', 'day_of_week', 'hour', 'minute']]

    x_train = pd.DataFrame(np.hstack([x_train, v_train]))
    x_test = pd.DataFrame(np.hstack([x_test, v_test]))

    print (x_train.shape)
    print (v_train.shape)

    return x_train, x_test, y_train, y_test




"""
Master function to return the vectorized training and test sets.
"""
def get_dataset():
    df = pd.read_pickle('ten_percent_sampled.pkl')
<<<<<<< HEAD
    # for j in range(1,df.shape[0]):
    #     if not wordnet.synsets(df["title"].iloc[j]):#Comparing if word is non-English
    #        df.drop(df.index[j], inplace=True)
    #
    # print(df.shape)




    #df.to_pickle('ten-percent-sampled.pkl')
    x_train, x_test, y_train, y_test = split_data(df)

    v_train, v_test = vectorize_text_two(x_train['title'], x_test['title'])
    #x_train = x_train[['day_of_year', 'day_of_week', 'hour', 'minute']]
    #x_test = x_test[['day_of_year', 'day_of_week', 'hour', 'minute']]

    #x_train = pd.DataFrame(np.hstack([x_train, v_train]))
    #x_test = pd.DataFrame(np.hstack([x_test, v_test]))

    return v_train, v_test, y_train, y_test
=======
    df = df.sample(frac=0.5)
    #df.to_pickle('ten-percent-sampled.pkl')
    x_train, x_test, y_train, y_test = split_data(df)

    v_train, v_test = vectorize_text(x_train['title'], x_test['title'])
    x_train = x_train[['day_of_year', 'day_of_week', 'hour', 'minute']]
    x_test = x_test[['day_of_year', 'day_of_week', 'hour', 'minute']]

    x_train = pd.DataFrame(np.hstack([x_train, v_train]))
    x_test = pd.DataFrame(np.hstack([x_test, v_test]))
    print x_train.shape
    print x_test
    return x_train, x_test, y_train, y_test
>>>>>>> f8cd2ac4c27183be16bf1e7d83b6bd581c317307



"""
Function that splits df into train and test sets based on a 80:20 split.
"""
def split_data(df):
    y = df['above_median']
    x = df.drop('score', axis = 1)
    # print y
    print x
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test




def vectorize_text_two(x_train,x_test):


    x_train = x_train.fillna("")
    count_vect = CountVectorizer(stop_words = 'english')
    text_train = count_vect.fit_transform(x_train)
    words = count_vect.get_feature_names()
    freqs = text_train.toarray().sum(axis=0)
    dict = {}
    for i in range(len(words)):
        dict[words[i]] = freqs[i]




    #A = count_vect.vocabulary_
    #topFive = dict(sorted(A.iteritems(), key=operator.itemgetter(1), reverse=True)[:100])
    a1_sorted_keys = sorted(dict, key=dict.get, reverse=True)
    for r in a1_sorted_keys:
        print r, dict[r]
    # tfidf_transformer = TfidfTransformer()
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    # print X_train_tfidf.shape


    text_test= count_vect.transform(x_test)
    # X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    # text_train, text_test = pd.DataFrame(text_train), pd.DataFrame(text_test)

    return text_train, text_test





"""
Function that vectorizes the text data in the dataset.
"""
def vectorize_text(x_train, x_test):
    vect = TV(strip_accents='unicode', analyzer='word')
    text_train = vect.fit_transform(x_train)
    text_test = vect.transform(x_test)
    # text_train, text_test = pd.DataFrame(text_train), pd.DataFrame(text_test)
    return text_train, text_test

def main():
    df = pd.read_pickle('aggregate.pkl')
    print df.shape
    train_set = df.iloc[0:int(df.shape[0]*0.8)]
    test_set = df.iloc[int(df.shape[0]*0.8):df.shape[0]-1]
    x_train = train_set.drop(columns = ['score','created_utc'])
    x_test = test_set.drop(columns = ['score','created_utc'])
    text_train, text_test = vectorize_text(x_train,x_test)
    print text_train.shape

    reg = random_forest(text_train,train_set['score'],text_test,test_set['score'])
    print reg.score(x_test, y_test)

if __name__ == '__main__':
    get_dataset()
