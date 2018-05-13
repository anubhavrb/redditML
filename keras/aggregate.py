import numpy as np
import pandas as pd
import json

from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
def vectorize_text(x_train, x_test):

    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ',lower=True, split=' ')
    tokenizer.fit_on_texts(x_train)
    # Tokenizers come with a convenient list of words and IDs
    dictionary = tokenizer.word_index
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


def read_dataset_from_file():
    df = pd.read_pickle('ten_percent_sampled.pkl')
    X_train, X_test, Y_train, Y_test = split_data(df)

    v_train, v_test = vectorize_text(X_train['title'], X_test['title'])
    x_train = X_train[['day_of_year', 'day_of_week', 'hour', 'minute']]
    x_test = X_test[['day_of_year', 'day_of_week', 'hour', 'minute']]

    x_train = pd.DataFrame(np.hstack([x_train, v_train]))
    x_test = pd.DataFrame(np.hstack([x_test, v_test]))

    print (x_train.shape)
    print (v_train.shape)
    return v_train, v_test, Y_train, Y_test
