import numpy as np 
import skimage.io 
import sys 

def read_data(file_path): 
    return np.load(file_path)

def normalize(X): 
    mu, sigma = statistics(X) 
    X = (X -  mu) / sigma 
    return X 

def denormalize(N, mu, sigma): 
    X = N * sigma + mu 
    return X 

def statistics(X): 
    mu = np.mean(X, axis=0) 
    sigma = np.std(X, axis=0) + 1e-10
    return mu, sigma 

def svd(X): 
    C = np.cov(X.T) 
    U, S, V = np.linalg.svd(C)
    return U, S, V 

def reduce_dimension(X, U, k): 
    return np.dot(X, U[:, :k]) 

def reconstruct(Z, U, k): 
    return np.dot(Z, U[:, :k].T) 

if __name__ == '__main__': 
    data_file_path = sys.argv[1] 
