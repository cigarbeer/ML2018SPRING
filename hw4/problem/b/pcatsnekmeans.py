import numpy as np 
from sklearn.decomposition import PCA 
from sklearn.preprocessing import scale 
from sklearn.manifold import TSNE 
from sklearn.cluster import KMeans 
import pandas as pd 
import sys 
import pickle 


if __name__ == '__main__': 
    file_name = sys.argv[1] 
    X = np.load(file_name) 
    X_n = scale(X) 
    pca = PCA(
        n_components=20,
        copy=True, 
        whiten=False, 
        svd_solver='full', 
        tol=0.0, 
        iterated_power='auto', 
        random_state=None 
    ) 
    pca.fit(X_n) 
    X_pca = pca.transform(X_n) 
    tsne = TSNE(
        n_components=2, 
        perplexity=30.0, 
        early_exaggeration=12.0, 
        learning_rate=200.0, 
        n_iter=1000, 
        n_iter_without_progress=300, 
        min_grad_norm=1e-07, 
        metric='euclidean', 
        init='random', 
        verbose=1, 
        random_state=None, 
        method='barnes_hut', 
        angle=0.5
    )
    tsne.fit(X_pca) 
    X_tsne = tsne.embedding_ 
    kmeans = KMeans(
        n_clusters=2, 
        init='k-means++',
        n_init=10, 
        max_iter=300, 
        tol=0.0001, 
        precompute_distances='auto', 
        verbose=1, 
        random_state=None, 
        copy_x=True, 
        n_jobs=-1, 
        algorithm='auto'
    ) 
    kmeans.fit(X_tsne) 
    with open('pca20.pickle', 'wb') as f:
        pickle.dump(pca, f) 
        f.close() 
    with open('tsne20to2.pickle', 'wb') as f: 
        pickle.dump(tsne, f) 
        f.close() 
    with open('kmeans2.pickle', 'wb') as f: 
        pickle.dump(kmeans, f) 
        f.close() 

    