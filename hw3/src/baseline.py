import numpy as np 
np.random.seed(0)
import pandas as pd 

from keras.models import Sequential 

from keras.layers import Conv2D 
from keras.layers import MaxPooling2D 
from keras.layers import Activation 
from keras.layers import BatchNormalization 
from keras.layers import Dropout 
from keras.layers import Flatten 
from keras.layers import Dense 

from keras.optimizers import Adam 

from keras.utils import np_utils 

from keras.callbacks import EarlyStopping 
from keras.callbacks import ModelCheckpoint 


INPUT_SHAPE = (48, 48, 1)
OUTPUT_SIZE = 7 
FLATTEN_SIZE = 48 * 48


def read_training_data(file_name):
    df = pd.read_csv(file_name) 
    label = df.label 
    feature = df.feature.apply(np.fromstring, sep=' ', dtype=np.float32)
    X = np.stack(feature, axis=0)
    y = np_utils.to_categorical(label)
    return X, y 

def read_testing_data(file_name):
    df = pd.read_csv(file_name) 
    ids = df.id.values
    feature = df.feature.apply(np.fromstring, sep=' ', dtype=np.float32)
    t = np.stack(feature, axis=0)
    return ids, t

def preprocess_data(X):
    # X = X / 255 
    mu = np.mean(X, axis=1) 
    sigma = np.std(X, axis=1) 
    X = ((X.T - mu) / sigma).T
    X = X.reshape((-1, *INPUT_SHAPE))
    return X 

def write_prediction(file_name, prediction):
    df = pd.DataFrame(columns=['label'], data=prediction)
    df.to_csv(file_name, index=True, index_label='id')
    return df 


def cnn_input(input_shape, filters, kernel_size):
    model = Sequential() 
    model.add(Conv2D(
        filters=filters, 
        kernel_size=kernel_size, 
        strides=(1, 1), 
        padding='same', 
        data_format='channels_last', 
        use_bias=True, 
        kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', 
        input_shape=input_shape 
    ))
    model.add(BatchNormalization())
    return model 


def cnn_hidden(model, filters, kernel_size, n_layers, dropout_rate):
    for n in range(n_layers):
        model.add(Conv2D(
            filters=filters, 
            kernel_size=kernel_size, 
            strides=(1, 1), 
            padding='same', 
            data_format='channels_last', 
            use_bias=True, 
            kernel_initializer='glorot_uniform', 
            bias_initializer='zeros'
        ))
        model.add(Activation('relu'))
        model.add(BatchNormalization()) 
    model.add(MaxPooling2D(
        pool_size=(2, 2), 
        strides=(2, 2), 
        padding='same', 
        data_format='channels_last' 
    ))
    model.add(Dropout(
        rate=dropout_rate
    ))
    return model 

def cnn_output(model, output_size, units, n_layers, dropout_rate): 
    model.add(Flatten())
    for n in range(n_layers):
        model.add(Dense(
            units=units, 
            use_bias=True, 
            kernel_initializer='glorot_uniform', 
            bias_initializer='zeros'
        ))
        model.add(Activation('relu'))
        model.add(Dropout(
            rate=dropout_rate
        ))
    model.add(Dense(
        units=output_size, 
        activation='softmax', 
        use_bias=True, 
        kernel_initializer='glorot_uniform', 
        bias_initializer='zeros'
    ))
    return model 

def compile_model(model):
    adam = Adam(lr=1e-4) 
    model.compile(
        optimizer=adam, 
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model 

def fit_model(model, X, y, epochs, batch_size, model_saving_path):
    callbacks = [
        ModelCheckpoint(model_saving_path+'weights.{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5', monitor='val_loss', verbose=1), 
        EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=3, verbose=1)
    ]
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, shuffle=True, callbacks=callbacks, verbose=1)
    return model 

def predict(model, t, batch_size): 
    prob = model.predict(t, batch_size=batch_size, verbose=1)
    pred = np.argmax(pred, axis=1)
    return pred 

def write_prediction(file_name, prediction):
    df = pd.DataFrame(columns=['label'], data=prediction)
    df.to_csv(file_name, index=True, index_label='id')
    return df 
