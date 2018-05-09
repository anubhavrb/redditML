import pandas as pd
from aggregate import read_dataset_from_file
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
def run_model():
    model = Sequential()
    x_train, x_test, y_train, y_test = read_dataset_from_file()

    # model.add(Dense(input_dim = 10000, units = 40,activation = 'relu'))

    model.add(Dense(x_train.shape[0], input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam')


if __name__ == '__main__':
    run_model()
