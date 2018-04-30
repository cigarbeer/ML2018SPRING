import sys 
import numpy as np 
import pandas as pd 
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import scale 
import pickle 


LAMBDA = 32 
EPOCH = 30000 
ETA = 1e-4 

def read_training_data(X_file, y_file):
    X = pd.read_csv(X_file) 
    y = pd.read_csv(y_file, header=None) 
    return X.values, y.values

class Ensemble: 
    def __init__(self):
        self.training_sets = None 
        self.models = None 
    
    def sample_training_sets(self, X, y, n_sets): 
        self.training_sets = []
        m, _ = X.shape 
        for n in range(n_sets):
            sample_idx = np.random.choice(m, size=m, replace=True) 
            self.training_sets.append((X[sample_idx], y[sample_idx])) 
        return 

    def train(self): 
        self.models = [] 
        for X, y in self.training_sets: 
            X_n = scale(X) 
            lr = LogisticRegression(C=1/LAMBDA, solver='sag', max_iter=EPOCH, verbose=1, n_jobs=-1) 
            lr.fit(X, y.flatten()) 
            self.models.append(lr) 
        return 

    def predict(self, t): 
        results = [] 
        threshold = len(self.models) / 2
        for (X, y), lr in zip(self.training_sets, self.models): 
            mu = np.mean(X, axis=0) 
            sigma = np.std(X, axis=0) 
            t = (t - mu) /sigma 
            results.append(lr.predict(t)) 
        results = np.array(results) 
        results = np.sum(results, axis=1) 
        pred = (results > threshold).astype(np.uint8) 
        return pred 

if __name__ == '__main__': 
    X_file = sys.argv[1] 
    y_file = sys.argv[2] 
    X, y = read_training_data(X_file, y_file) 
    ensemble = Ensemble() 
    ensemble.sample_training_sets(X, y, n_sets=5) 
    ensemble.train() 
    with open('ensemble.pickle', 'wb') as f:
        pickle.dump(ensemble, f) 
        f.close() 
    
