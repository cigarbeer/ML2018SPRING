import sys 
import os 
import pickle 

import numpy as np 
import pandas as pd  

from keras.models import Model 
from keras.models import load_model 

from keras.layers import Input 
from keras.layers import Dense 
from keras.layers import Embedding 
from keras.layers import Dot 
from keras.layers import Add 
from keras.layers import Flatten 
from keras.layers import Concatenate 
from keras.layers import Dropout 

from keras.callbacks import EarlyStopping 
from keras.callbacks import ModelCheckpoint 

from setting import * 


def read_training_data(path): 
    df = pd.read_csv(path) 
    df = df.sample(frac=1, replace=False) 
    uid = df.UserID.astype(np.int32) 
    mid = df.MovieID.astype(np.int32) 
    rating = df.Rating.astype(np.float32) 
    return uid, mid, rating 

def normalize_rating(rating): 
    mu = np.mean(rating) 
    sigma = np.std(rating) 
    np.save(RATING_STATISTICS_PATH, [mu, sigma]) 
    rating_normalized = (rating - mu) / sigma 
    return rating_normalized 

def get_matrix_factorization_model(matrix_shape, latent_dimension): 
    m, n = matrix_shape 
    m = m + 1
    n = n + 1 

    input_m = Input(shape=(1,)) 
    input_n = Input(shape=(1,)) 

    embedding_m = Embedding(input_dim=m, output_dim=latent_dimension)(input_m)
    embedding_n = Embedding(input_dim=n, output_dim=latent_dimension)(input_n)

    bias_m = Embedding(input_dim=m, output_dim=1)(input_m) 
    bias_n = Embedding(input_dim=n, output_dim=1)(input_n) 

    flatten_embedding_m = Flatten()(embedding_m) 
    flatten_embedding_n = Flatten()(embedding_n) 
    flatten_bias_m = Flatten()(bias_m) 
    flatten_bias_n = Flatten()(bias_n) 

    dot_mn = Dot(axes=1, normalize=False)([flatten_embedding_m, flatten_embedding_n]) 

    output = Add()([dot_mn, flatten_bias_m, flatten_bias_n]) 

    model = Model(inputs=[input_m, input_n], outputs=[output]) 

    model.compile(optimizer='rmsprop', loss='mse') 

    model.summary()

    return model 

def train(model, X, y, batch_size, epochs, validation_split, model_checkpoint_path): 
    callbacks = [
        EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=1), 
        ModelCheckpoint(filepath=model_checkpoint_path, monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1) 
    ]

    model.fit(
        x=X, 
        y=y, 
        batch_size=batch_size, 
        epochs=epochs,
        callbacks=callbacks,
        validation_split=validation_split,
        shuffle=True, 
        verbose=1
    ) 

    return model 

def predict(model, t, batch_size): 
    pred = model.predict(t, batch_size=batch_size, verbose=1) 
    mu, sigma = np.load(RATING_STATISTICS_PATH) 
    pred = pred * sigma + mu 
    return pred 

def write_prediction(pred, path): 
    df = pd.DataFrame(columns=['Rating'], data=pred) 
    df.index = df.index + 1 
    df.to_csv(path, index=True, index_label='TestDataID', float_format='%.3f') 
    return df 

def read_testing_data(path): 
    df = pd.read_csv(path) 
    tuid = df.UserID 
    tmid = df.MovieID 
    return tuid, tmid  

def load_matrix_factorization_model(): 
    return load_model(MATRIX_FACTORIZATION_CHECKPOINT_PATH) 


def get_best_model(matrix_shape, latent_dimension): 
    m, n = matrix_shape 
    m = m + 1
    n = n + 1 

    input_m = Input(shape=(1,)) 
    input_n = Input(shape=(1,)) 

    embedding_m = Embedding(input_dim=m, output_dim=latent_dimension)(input_m)
    embedding_n = Embedding(input_dim=n, output_dim=latent_dimension)(input_n) 

    flatten_embedding_m = Flatten()(embedding_m) 
    flatten_embedding_n = Flatten()(embedding_n)

    concat_flatten_embedding_mn = Concatenate()([flatten_embedding_m, flatten_embedding_n]) 

    dense_1 = Dense(units=latent_dimension, activation='selu')(concat_flatten_embedding_mn) 
    dropout_1 = Dropout(rate=0.5)(dense_1)
    dense_2 = Dense(units=latent_dimension, activation='selu')(dropout_1)
    dropout_2 = Dropout(rate=0.5)(dense_2)
    dense_3 = Dense(units=latent_dimension, activation='selu')(dropout_2) 
    dropout_3 = Dropout(rate=0.5)(dense_3)

    output = Dense(units=1, activation='relu')(dropout_3) 

    model = Model(inputs=[input_m, input_n], outputs=[output]) 

    model.compile(optimizer='rmsprop', loss='mse') 
    model.summary() 

    return model  
