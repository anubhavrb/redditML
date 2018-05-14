import pandas as pd
from aggregate import read_dataset_from_file
from keras.models import Sequential
from keras.layers import Dense, Activation,Flatten
from keras import optimizers
from keras import backend as K
from keras import metrics
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import graphviz
from keras.layers import Embedding
from keras.utils import plot_model
from aggregate import get_dataset_binary
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy

def coeff_determination(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
def run_model():
    model = Sequential()
    x_train, x_test, y_train, y_test = get_dataset_binary()

    print(x_train['title'])
    print(y_train.values)
    vocab_size = 40000
    max_length = 50
    embedding_space = 25
    encoded_docs = [one_hot(x_train['title'].iloc[d], vocab_size) for d in range(x_train.shape[0])]
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    print(padded_docs)
    print("Y_train is:")
    print(y_train)
    input_dim =  x_train.shape[1]
    output_dim = 32

    model.add(Embedding(vocab_size, embedding_space, input_length=max_length))
    model.add(Flatten())
    model.add(Dense(10, kernel_initializer='normal',activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])
    print(model.summary())



<<<<<<< HEAD
    history = model.fit(padded_docs,y_train.values, batch_size=128, epochs = 15)
    plot_model(model, to_file='model.png')

    print(history.history.keys())

=======
    history = model.fit(padded_docs,y_train.values, batch_size=128, epochs = 2)
    plot_model(model, to_file='model.png')
    # model.evaluate(x_test, y_test, batch_size=128)
    loss, accuracy = model.evaluate(padded_docs, y_train.values, verbose=0)
    print('Accuracy: %f' % (accuracy*100))
>>>>>>> f8cd2ac4c27183be16bf1e7d83b6bd581c317307
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'loss'], loc='upper left')
    plt.show()


    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    loss,accuracy = model.evaluate(padded_docs, y_train.values, verbose=0)
    print('Accuracy on new: %f' % (accuracy*100))
    encoded_docs_test = [one_hot(x_test['title'].iloc[d], vocab_size) for d in range(x_test.shape[0])]
    padded_docs_test = pad_sequences(encoded_docs_test, maxlen=max_length, padding='post')
    loss,accuracy = model.evaluate(padded_docs_test, y_test.values, verbose=0)
    print('Accuracy: %f' % (accuracy*100))



    # # model.add(Embedding(20,
    # #                 embedding_dims,
    # #                 input_length=maxlen))
    # # model.add(GlobalAveragePooling1D())
    # # Add in regulations 64/128
    # model.add(Dense(100, input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu'))
    # model.add(Dense(100, input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu'))
    # model.add(Dense(1, kernel_initializer='normal'))


    #







if __name__ == '__main__':
    run_model()
