from keras.models import load_model 
from sklearn.manifold import TSNE 
import pandas as pd 
import numpy as np 
import sys 
import pickle 
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt 

def save_object(obj, path): 
    with open(path, mode='wb') as f: 
        pickle.dump(obj, f) 
        f.close() 
    return 

def load_object(path): 
    obj = None 
    with open(path, mode='rb') as f: 
        obj = pickle.load(f) 
        f.close() 
    return obj

def read_movies(path): 
    df = pd.read_table(path, sep='::') 
    df.Genres = df.Genres.str.split('|') 
    return df 

def get_movies_by_genres(movies, genres): 
    mids = [] 
    for mid, row in movies.iterrows(): 
        if genres in row['Genres']: 
            mids.append(row['movieID']) 
    return mids 

# from keras.models import Model 
# model =  Model()
if __name__ == '__main__': 
    model_path = sys.argv[1] 
    movies_path = sys.argv[2] 

    model = load_model(model_path) 

    movies = read_movies(movies_path)

    movie_emb = model.get_layer(name='embedding_2').get_weights()[0] 

    comedy_mids = get_movies_by_genres(movies, 'Comedy')
    romance_mids = get_movies_by_genres(movies, 'Romance')
    action_mids = get_movies_by_genres(movies, 'Action') 

    comedy = movie_emb[comedy_mids] 
    romance = movie_emb[romance_mids] 
    action = movie_emb[action_mids] 

    concat = np.concatenate([comedy, romance, action]) 

    tsne = TSNE(n_components=2, verbose=1)  

    tsne.fit(concat)  

    save_object(tsne, './tsne.pickle') 

    concat_2d = tsne.embedding_ 

    comedy_len = len(comedy) 
    romance_len = len(romance) 
    action_len = len(action) 

    comedy_2d = concat_2d[:comedy_len] 
    romance_2d = concat_2d[comedy_len:comedy_len+romance_len] 
    action_2d = concat_2d[comedy_len+romance_len:] 

    fig, ax = plt.subplots()
    ax.scatter(comedy_2d[:, 0], comedy_2d[:, 1], label='Comedy')
    ax.scatter(romance_2d[:, 0], romance_2d[:, 1], label='Romance')
    ax.scatter(action_2d[:, 0], action_2d[:, 1], label='Action')
    ax.legend()
    fig.savefig('./tsne.png') 
    


