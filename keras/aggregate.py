import numpy as np
import pandas as pd
import json

from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import operator
from nltk.corpus import wordnet

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def vectorize_text(x_train, x_test):

    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ',lower=True, split=' ')
    tokenizer.fit_on_texts(x_train)
    # Tokenizers come with a convenient list of words and IDs
    dictionary = tokenizer.word_index

    # i = 0
    # for key, value in sorted(dictionary.iteritems(), key=lambda (k,v): (v,k)):
    #     i += 1
    #     if(i < 10):
    #         print "%s: %s" % (key, value)

    # Let's save this out so we can use it later
    with open('dictionary.json', 'w') as dictionary_file:
        json.dump(dictionary, dictionary_file)

    text_train = tokenizer.texts_to_matrix(x_train)
    text_test= tokenizer.texts_to_matrix(x_test)



    print (text_train)


    print('text_train shape:', text_train.shape)
    print('text_test shape:', text_test.shape)

    # text_train, text_test = pd.DataFrame(text_train), pd.DataFrame(text_test)
    return text_train, text_test



def split_data(df):
    y = df['score']
    x = df.drop('score', axis = 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test
def get_dataset():
    df = pd.read_pickle('ten_percent_sampled.pkl')
    df = df.reset_index()

    X_train, X_test, Y_train, Y_test = split_data(df)
    return X_train,X_test,Y_train,Y_test

def read_dataset_from_file():
    df = pd.read_pickle('ten_percent_sampled.pkl')
    df = df.reset_index()
    # print df.shape
    # print df
    #
    # for j in range(0,df.shape[0]-1):
    #     if not wordnet.synsets(df['title'].iloc[j]):
    #         df.drop([j],inplace=True)
    #     sys.stdout.write("Deletion Progress: %d/225450   \r"%(j))
    #     sys.stdout.flush()
    #
    #
    # df.to_pickle("ten_percent_engish_only.pkl")

    X_train, X_test, Y_train, Y_test = split_data(df)
    
    v_train, v_test = vectorize_text(X_train['title'], X_test['title'])
    x_train = X_train[['day_of_year', 'day_of_week', 'hour', 'minute']]
    x_test = X_test[['day_of_year', 'day_of_week', 'hour', 'minute']]

    x_train = pd.DataFrame(np.hstack([x_train, v_train]))
    x_test = pd.DataFrame(np.hstack([x_test, v_test]))

    # print (x_train.shape)
    # print (v_train.shape)
    return v_train, v_test, Y_train, Y_test
