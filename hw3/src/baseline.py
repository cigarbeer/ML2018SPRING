SEED = 0 
import numpy as np 
np.random.seed(SEED)
import pandas as pd 


from keras.preprocessing.image import ImageDataGenerator 

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

VALIDATION_SPLIT = 0.2 
BATCH_SIZE = 128 

INPUT_SHAPE = (48, 48, 1)
OUTPUT_SIZE = 7 
FLATTEN_SIZE = 48 * 48


def read_raw_training_data(file_name):
    df = pd.read_csv(file_name) 
    label = df.label 
    feature = df.feature.apply(np.fromstring, sep=' ', dtype=np.float32)
    X = np.stack(feature, axis=0)
    y = np_utils.to_categorical(label)
    return X, y 

def read_raw_testing_data(file_name):
    df = pd.read_csv(file_name) 
    ids = df.id.values
    feature = df.feature.apply(np.fromstring, sep=' ', dtype=np.float32)
    t = np.stack(feature, axis=0)
    return ids, t

def read_selected_training_data(X_file, y_file):
    return np.load(X_file), np.load(y_file)  

def preprocess_training_data(X):
    X = samplewise_normalization(X) 
    X = featurewise_normalize(X) 
    X = X.reshape((-1, *INPUT_SHAPE))
    return X 

def preprocess_testing_data(t):
    return preprocess_training_data(t) 

def split_validation_set(X, y, rate):
    m, *n = X.shape 
    n_train = int((1 - rate) * m)
    X_train = X[:n_train]
    y_train = y[:n_train] 
    X_val = X[n_train:]
    y_val = y[n_train:]
    return (X_train, y_train), (X_val, y_val) 

def get_training_data_generator(X):
    training_data_generator = ImageDataGenerator(
        samplewise_center=False, 
        samplewise_std_normalization=False, 
        featurewise_center=False, 
        featurewise_std_normalization=False, 
        zca_whitening=False, 
        zca_epsilon=1e-06, 
        rotation_range=10.0, 
        width_shift_range=0.05, 
        height_shift_range=0.05, 
        shear_range=0.0, 
        zoom_range=0.05, 
        channel_shift_range=0.0, 
        fill_mode='nearest', 
        cval=0.0, 
        horizontal_flip=True, 
        vertical_flip=False, 
        rescale=None, 
        preprocessing_function=None, 
        data_format='channels_last', 
    )
    training_data_generator.fit(X, augment=False, rounds=1, seed=SEED)
    return training_data_generator


def get_testing_data_generator(t):
    testing_data_generator = ImageDataGenerator(
        samplewise_center=False, 
        samplewise_std_normalization=False, 
        featurewise_center=False, 
        featurewise_std_normalization=False, 
        zca_whitening=False, 
        zca_epsilon=1e-06, 
        rotation_range=0.0, 
        width_shift_range=0.0, 
        height_shift_range=0.0, 
        shear_range=0.0, 
        zoom_range=0.0, 
        channel_shift_range=0.0, 
        fill_mode='nearest', 
        cval=0.0, 
        horizontal_flip=False, 
        vertical_flip=False, 
        rescale=None, 
        preprocessing_function=None, 
        data_format='channels_last'
    )
    testing_data_generator.fit(t, augment=False, rounds=1, seed=SEED)
    return testing_data_generator

def get_validation_data_generator(V):
    return get_testing_data_generator(V)

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
    model.add(Activation('relu'))
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

def cnn_to_nn(model): 
    model.add(Flatten())
    return model 

def nn(model, units, n_layers, dropout_rate):
    for n in range(n_layers):
        model.add(Dense(
            units=units,
            use_bias=True, 
            kernel_initializer='glorot_uniform', 
            bias_initializer='zeros'
        ))
        model.add(Activation('relu'))
        model.add(BatchNormalization()) 
        model.add(Dropout(
            rate=dropout_rate
        ))
    return model 

def nn_output(model, output_size):
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

def fit_generator(model, X, y, epochs, batch_size, model_saving_path):
    callbacks = [
        ModelCheckpoint(model_saving_path+'weights.{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5', monitor='val_loss', verbose=1), 
        EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=4, verbose=1)
    ]

    (X_train, y_train), (X_val, y_val) = split_validation_set(X, y, VALIDATION_SPLIT) 
    m_train, *n = X_train.shape 
    m_val, *n = X_val.shape 

    # training_data_generator = get_training_data_generator(X_train).flow(X_train, y_train, batch_size=batch_size, shuffle=True, seed=SEED, save_to_dir='./aug/', save_prefix='train/', save_format='png')
    # validation_data_generator = get_validation_data_generator(X_train).flow(X_val, y_val, batch_size=batch_size, shuffle=True, seed=SEED, save_to_dir='./aug/', save_prefix='val/', save_format='png')
    training_data_generator = get_training_data_generator(X_train).flow(X_train, y_train, batch_size=batch_size, shuffle=True, seed=SEED)
    validation_data_generator = get_validation_data_generator(X_train).flow(X_val, y_val, batch_size=batch_size, shuffle=True, seed=SEED)

    model.fit_generator(
        generator=training_data_generator, 
        steps_per_epoch=int(m_train/batch_size)+1, 
        epochs=epochs, 
        verbose=1, 
        callbacks=callbacks, 
        validation_data=validation_data_generator, 
        validation_steps=int(m_val/batch_size)+1, 
        class_weight=None, 
        max_queue_size=10, 
        workers=1, 
        use_multiprocessing=False, 
        initial_epoch=0
    )
    return model 

# def predict(model, generator, t, batch_size, )


def fit_model(model, X, y, epochs, batch_size, model_saving_path):
    callbacks = [
        ModelCheckpoint(model_saving_path+'weights.{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5', monitor='val_loss', verbose=1), 
        EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=4, verbose=1)
    ]
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, shuffle=True, callbacks=callbacks, verbose=1)
    return model 

# def predict(model, t, batch_size): 
#     prob = model.predict(t, batch_size=batch_size, verbose=1)
#     pred = np.argmax(prob, axis=1)
#     return pred 

def featurewise_normalize(X):
    mu = np.mean(X, axis=0) 
    sigma = np.std(X, axis=0)
    X = (X - mu) / sigma 
    return X 

def samplewise_normalization(X):
    mu = np.mean(X, axis=1) 
    sigma = np.std(X, axis=1)
    X = ((X.T - mu) / sigma).T  
    return X 
