import numpy as np
import pandas as pd
import ast
import os
import glob, os
import json
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.feature_extraction.text import TfidfTransformer as TF
def parse_csv():
    df_2016 = pd.read_csv("2016.csv")
    df_2017 = pd.read_csv("2017.csv")
    df = pd.concat([df_2016,df_2017])
    return df

def vectorize_text(x_train, x_test):

   vectorizer = TF(strip_accents='unicode', analyzer = 'word')
   text_train = vectorizer.fit_transform(x_train).toarray()
   text_test = vectorizer.transform(x_test).toarray()
   text_train,text_test = pd.DataFrame(text_train),pd.DataFrame(text_test)
   return text_train, text_test


def main():
    df = parse_csv()
    df = df.sample(frac=1)
    df = df.reset_index(drop = True)
    train_set = df.iloc[0:int(df.shape[0]*0.8)]
    test_set = df.iloc[int(df.shape[0]*0.8):df.shape[0]-1]
    x_train = train_set.drop(columns = ['score','created_utc'])
    x_test = test_set.drop(columns = ['score','created_utc'])

    text_train, text_test = vectorize_text(x_train,x_test)
    print (text_train)



if __name__ == '__main__':
    main()
