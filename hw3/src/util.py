import numpy as np 
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


INPUT_SHAPE = (48, 48, 1)
OUTPUT_SIZE = 7 
FLATTEN_SIZE = 48 * 48


def read_training_data(file_name):
    df = pd.read_csv(file_name) 
    label = df.label 
    feature = df.feature.apply(np.fromstring, sep=' ', dtype=np.float16)
    X = np.stack(feature, axis=0).reshape((-1, *INPUT_SHAPE))
    y = np_utils.to_categorical(label)
    return X, y 



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


def cnn_hidden(model, filters, kernel_size):
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
    model.add(BatchNormalization()) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(
        pool_size=(2, 2), 
        strides=(2, 2), 
        padding='same', 
        data_format='channels_last' 
    ))
    # model.add(Dropout(
    #     rate=0.5
    # ))
    return model 

def cnn_output(model, output_size, units): 
    model.add(Flatten())
    model.add(Dense(
        units=units, 
        activation='relu',
        use_bias=True, 
        kernel_initializer='glorot_uniform', 
        bias_initializer='zeros'
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