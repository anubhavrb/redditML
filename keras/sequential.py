import pandas as pd
from aggregate import read_dataset_from_file
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras import backend as K
from keras import metrics
import graphviz
from keras.layers import Embedding
from keras.utils import plot_model

def coeff_determination(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
def run_model():
    model = Sequential()
    x_train, x_test, y_train, y_test = read_dataset_from_file()
"""
    model.add(Dense(x_train.shape[0], input_dim=x_train.shape, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
"""

    max_features = 20000
    embedding_dims = 50
    # model.add(Dense(input_dim = 10000, units = 40,activation = 'relu'))
    # model.add(Embedding(input_dim=x_train.shape[1], output_dim=10))
    model.add(Embedding(20,
                    embedding_dims,
                    input_length=maxlen))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(10, input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=[coeff_determination])
    model.fit(x_train,y_train, batch_size=128, epochs = 10)

    plot_model(model, to_file='model.png')



if __name__ == '__main__':
    run_model()
