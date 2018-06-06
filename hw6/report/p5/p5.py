import sys 
import os 
import numpy as np 
import pandas as pd 

from keras.models import Model 
from keras.models import load_model 

from keras.utils import to_categorical 

from keras.layers import Input
from keras.layers import Dense 
from keras.layers import Embedding 
from keras.layers import Flatten 
from keras.layers import Concatenate 
from keras.layers import Dropout 
from keras.layers import BatchNormalization 

from keras.callbacks import EarlyStopping 
from keras.callbacks import ModelCheckpoint 


def read_users(path): 
    df = pd.read_table(path, sep='::') 
    # df = df.drop(columns=['Zip-code']) 
    df = df.sort_values(by=['UserID']) 
    df.Gender[df.Gender == 'M'] = 1 
    df.Gender[df.Gender == 'F'] = 0 
    gender = to_categorical(df.Gender) 
    user_id = df.UserID.values.reshape((-1, 1))
    age = df.Age.values.reshape((-1, 1)) 
    occupation = to_categorical(df.Occupation) 
    users = np.concatenate([user_id ,gender, age, occupation], axis=1) 
    m, n = users.shape
    users = np.concatenate([np.zeros((1, n)), users]) 
    return users 

def read_movies(path): 
    df = pd.read_table(path, sep='::') 
    # df = df.drop(columns=['Title']) 
    df = df.sort_values(by=['movieID']) 
    genres = np.unique(np.concatenate(df['Genres'].str.split('|').values)) 
    df['Genres'] = df['Genres'].str.split('|') 
    genres_dct = {}
    for i, g in enumerate(genres): 
        genres_dct[g] = i 
    genres_onehot = np.zeros((df['movieID'].max()+1, len(genres_dct)+1)) 
    for index, row in df.iterrows():
        for g in row['Genres']: 
            genres_onehot[row['movieID']][0] = row['movieID'] 
            genres_onehot[row['movieID']][genres_dct[g]+1] = 1.0 
    return genres_onehot, genres_dct  

def read_training(path): 
    df = pd.read_csv(path) 
    df = df.sample(frac=1, replace=False) 
    uid = df.UserID.astype(np.int32) 
    mid = df.MovieID.astype(np.int32) 
    rating = df.Rating.astype(np.float32) 
    return uid, mid, rating 

def make_training(uid, mid, users, movies): 
    train = [] 
    for u, m in zip(uid, mid): 
        train.append(np.concatenate([users[u], movies[m]])) 
    return np.stack(train) 

def get_model(input_shape):
    input_layer = Input(shape=input_shape) 
    x = Dense(256, activation='selu')(input_layer) 
    x = BatchNormalization()(x) 
    x = Dropout(0.5)(x) 
    x = Dense(128, activation='selu')(x) 
    x = BatchNormalization()(x) 
    x = Dropout(0.5)(x)
    x = Dense(64, activation='selu')(x) 
    x = BatchNormalization()(x) 
    x = Dropout(0.5)(x)
    x = Dense(128, activation='selu')(x) 
    x = BatchNormalization()(x) 
    x = Dropout(0.5)(x)
    x = Dense(256, activation='selu')(x) 
    x = BatchNormalization()(x) 
    x = Dropout(0.5)(x)
    output_layer = Dense(1, activation='relu')(x) 

    model = Model(inputs=[input_layer], outputs=[output_layer]) 
    model.compile(optimizer='nadam', loss='mse') 
    model.summary()
    return model 

def train(model, X, y): 
    callbacks = [ 
        EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=1), 
        ModelCheckpoint(filepath='./p5.hdf5', monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1) 
    ]
    model.fit(
        x=X, 
        y=y, 
        batch_size=512, 
        epochs=100, 
        callbacks=callbacks, 
        validation_split=0.1, 
        shuffle=True, 
        verbose=1 
    ) 
    return model 

def read_testing(path): 
    df = pd.read_csv(path) 
    tuid = df.UserID 
    tmid = df.MovieID 
    return tuid, tmid   

def make_testing(tuid, tmid, users, movies): 
    test = [] 
    for u, m in zip(tuid, tmid):  
        test.append(np.concatenate([[tuid], users[u], [tmid], movies[m]])) 
    return np.stack(test) 

def predict(model, t): 
    pred = model.predict(t) 
    return pred 

def write_prediction(pred, path): 
    df = pd.DataFrame(columns=['Rating'], data=pred) 
    df.index = df.index + 1 
    df.to_csv(path, index=True, index_label='TestDataID', float_format='%.3f') 
    return df


def main(): 
    users = read_users(sys.argv[1]) 
    movies, movies_dct = read_movies(sys.argv[2]) 
    uid, mid, rating = read_training(sys.argv[3]) 
    X = make_training(uid, mid, users, movies) 
    model = get_model(X[0].shape) 
    train(model, X, rating) 
    model = load_model('./p5.hdf5') 
    tuid, tmid = read_testing(sys.argv[4]) 
    t = make_testing(tuid, tmid, users, movies) 
    pred = predict(model, test) 
    write_prediction(pred, './p5.csv') 

    