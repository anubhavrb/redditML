import pandas as pd
from aggregate import read_dataset_from_file
from keras.models import Sequential
from keras.layers import Dense, Activation
def run_model():
    model = Sequential()
    x_train, x_test, y_train, y_test = read_dataset_from_file()
    model.add(Dense(23050,input_shape =(23050,)))
    model.add(Activation('relu'))
    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    model.fit(x_train,x_test)

if __name__ == '__main__':
    run_model()
