SEED = 0 
import numpy as np 
np.random.seed(SEED)
import pandas as pd 

from keras.preprocessing.image import ImageDataGenerator 

from keras.models import Sequential 

from keras.layers import Reshape 
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

import sys 
import setting as st 

VALIDATION_SPLIT = 0.1 
BATCH_SIZE = 128
FULLY_CONNECTED_UNIT_NUM = 512
FLATTEN_IMAGE_SIZE = 48 * 48
FLATTEN_IMAGE_SHAPE = (FLATTEN_IMAGE_SIZE,)
IMAGE_SHAPE = (48, 48, 1)
OUTPUT_CLASSES_NUM = 7 


def read_raw_training_data(file_name):
    df = pd.read_csv(file_name) 
    label = df.label 
    feature = df.feature.apply(np.fromstring, sep=' ', dtype=np.float64)
    X = np.stack(feature, axis=0)
    y = np_utils.to_categorical(label, num_classes=OUTPUT_CLASSES_NUM)
    m, n = X.shape 
    rand_idx = np.random.permutation(np.arange(m)) 
    X = X[rand_idx] 
    y = y[rand_idx] 
    return X / 255, y 

def read_raw_testing_data(file_name):
    df = pd.read_csv(file_name) 
    ids = df.id.values
    feature = df.feature.apply(np.fromstring, sep=' ', dtype=np.float32)
    t = np.stack(feature, axis=0)
    return ids, t / 255

def read_selected_training_data(X_file, y_file):
    return np.load(X_file), np.load(y_file)  

def preprocess_training_data(X):
    X = samplewise_normalization(X) 
    X = X.reshape((-1, *IMAGE_SHAPE))
    return X, 0, 0  

def preprocess_testing_data(t, mu, sigma):
    t = samplewise_normalization(t) 
    t = t.reshape((-1, *IMAGE_SHAPE))
    return t 

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
        # samplewise_center=False, 
        # samplewise_std_normalization=False, 
        # featurewise_center=False, 
        # featurewise_std_normalization=False, 
        rotation_range=20.0, 
        width_shift_range=0.1, 
        height_shift_range=0.1, 
        horizontal_flip=True, 
        # zca_whitening=False, 
        # zca_epsilon=1e-06, 
        # shear_range=0.0, 
        # zoom_range=0, 
        # channel_shift_range=0.0, 
        # fill_mode='nearest', 
        # cval=0.0, 
        # vertical_flip=False, 
        # rescale=None, 
        # preprocessing_function=None, 
        data_format='channels_last', 
    )
    # training_data_generator.fit(X, augment=False, rounds=1, seed=SEED)
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
    # testing_data_generator.fit(t, augment=False, rounds=1, seed=SEED)
    return testing_data_generator

def get_validation_data_generator(V):
    return get_testing_data_generator(V)

def write_prediction(file_name, prediction):
    df = pd.DataFrame(columns=['label'], data=prediction)
    df.to_csv(file_name, index=True, index_label='id')
    return df 

def input_block(model, input_shape, output_shape):
    model.add(Reshape(target_shape=output_shape, input_shape=input_shape))
    return model 

def cnn_block(model, filters, kernel_size, n_layers, dropout_rate):
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
        model.add(Activation('selu'))
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

def nn_block(model, units, n_layers, dropout_rate):
    for n in range(n_layers):
        model.add(Dense(
            units=units,
            use_bias=True, 
            kernel_initializer='glorot_uniform', 
            bias_initializer='zeros'
        ))
        model.add(Activation('selu'))
        model.add(BatchNormalization()) 
        model.add(Dropout(
            rate=dropout_rate
        ))
    return model 

def output_block(model, output_shape):
    model.add(Dense(
        units=output_shape, 
        activation='softmax', 
        use_bias=True, 
        kernel_initializer='glorot_uniform', 
        bias_initializer='zeros'
    ))
    return model 

def net(input_shape, output_shape):
    model = Sequential()
    model = input_block(model, input_shape=input_shape, output_shape=IMAGE_SHAPE)

    model = cnn_block(model, filters=64, kernel_size=(3, 3), n_layers=2, dropout_rate=0.5) 
    model = cnn_block(model, filters=128, kernel_size=(3, 3), n_layers=2, dropout_rate=0.5) 
    model = cnn_block(model, filters=256, kernel_size=(3, 3), n_layers=2, dropout_rate=0.5) 

    model.add(Flatten())

    model = nn_block(model, units=FULLY_CONNECTED_UNIT_NUM, n_layers=3, dropout_rate=0.5) 

    model = output_block(model, output_shape=output_shape)

    return model 

def compile_model(model):
    model.compile(
        optimizer='nadam', 
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model 

def fit_generator(model, X, y, epochs, batch_size, model_saving_path):
    callbacks = [
        ModelCheckpoint(model_saving_path, monitor='val_loss', save_best_only=True, verbose=1), 
        EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=40, verbose=1)
    ]

    (X_train, y_train), (X_val, y_val) = split_validation_set(X, y, VALIDATION_SPLIT) 
    m_train, *n = X_train.shape 
    m_val, *n = X_val.shape 

    training_data_generator = get_training_data_generator(X_train).flow(X_train, y_train, batch_size=batch_size, shuffle=True, seed=SEED)

    model.fit_generator(
        generator=training_data_generator, 
        steps_per_epoch=int(m_train/batch_size), 
        epochs=epochs, 
        verbose=1, 
        callbacks=callbacks, 
        validation_data=(X_val, y_val), 
        validation_steps=None, 
        class_weight=None, 
        max_queue_size=10, 
        workers=1, 
        use_multiprocessing=False, 
        initial_epoch=0
    )
    return model 

def predict(model, t, batch_size): 
    prob = model.predict(t, batch_size=batch_size, verbose=1)
    pred = np.argmax(prob, axis=1)
    return pred 

def featurewise_normalize(X):
    mu = np.mean(X, axis=0) 
    sigma = np.std(X, axis=0)
    sigma = np.nan_to_num(sigma)
    X = (X - mu) / sigma 
    X = np.nan_to_num(X) 
    return X, mu, sigma 

def samplewise_normalization(X):
    mu = np.mean(X, axis=1) 
    sigma = np.std(X, axis=1) 
    sigma = np.nan_to_num(sigma)
    X = ((X.T - mu) / sigma).T  
    X = np.nan_to_num(X) 
    return X 

def save_statistics(s_path, mu, sigma):
    np.save(s_path, np.array([mu, sigma])) 
    return 

def load_statistics(s_path):
    return np.load(s_path)


if __name__ == '__main__':
    data_file = sys.argv[1] 
    X, y = read_raw_training_data(data_file) 
    X, mu, sigma = preprocess_training_data(X)
    save_statistics(st.STATISTICS_PATH, mu, sigma) 
    model = net(input_shape=IMAGE_SHAPE, output_shape=OUTPUT_CLASSES_NUM)
    model = compile_model(model)
    model.summary() 
    model = fit_generator(model, X, y, epochs=200, batch_size=BATCH_SIZE, model_saving_path=st.MODEL_PATH)
    model.save(st.MODEL_PATH)
    # idx, t = read_raw_testing_data('../../dataset/test.csv')
    # t = preprocess_testing_data(t, mu, sigma) 
    # pred = predict(model, t, batch_size=BATCH_SIZE)
    # write_prediction('pred.csv', pred) 
