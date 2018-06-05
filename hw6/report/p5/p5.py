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

from keras.callbacks import EarlyStopping 
from keras.callbacks import ModelCheckpoint 


def read_users(path): 
    df = pd.read_table(path, sep='::') 
    df = df.drop(columns=['Zip-code']) 
    df = df.sort_values(by=['UserID']) 
    df.Gender[df.Gender == 'M'] = 1 
    df.Gender[df.Gender == 'F'] = 0 
    gender = to_categorical(df.Gender) 
    user_id = df.UserID
    age = df.Age.reshape((-1, 1)) 
    occupation = to_categorical(df.Occupation) 
    users = np.concatenate([user_id, gender, age, occupation]) 
    return users 

def read_movies(path): 
    df = pd.read_table(path, sep='::') 
    df = df.drop(columns=['Title']) 
    df = df.sort_values(by=['movieID']) 
    genres = np.unique(np.concatenate(df['Genres'].str.split('|').values)) 
    df['Genres'] = df['Genres'].str.split('|') 
    genres_dct = {}
    for i, g in enumerate(genres): 
        genres_dct[g] = i 
    genres_onehot = np.zeros((df['movieID'].max()+1, len(genres_dct))) 
    for index, row in df.iterrows():
        for g in row['Genres']: 
            genres_onehot[row['movieID']][genres_dct[g]] = 1.0 
    return genres 



    
    