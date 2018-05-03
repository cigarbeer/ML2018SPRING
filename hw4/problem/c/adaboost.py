import sys 
import numpy as np 
import pandas as pd 
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import scale 
import pickle 


LAMBDA = 32 
EPOCH = 10000
ETA = 1e-4 

def read_training_data(X_file, y_file):
    X = pd.read_csv(X_file) 
    y = pd.read_csv(y_file, header=None) 
    return X.values, y.values.flatten()

def read_testing_data(t_file):
    t = pd.read_csv(t_file) 
    return t.values

def write_prediction(prediction_path, pred):
    df = pd.DataFrame(pred, columns=['label'])
    df.index = df.index + 1
    df.to_csv(prediction_path, index_label='id')
    return df 

class Adaboost: 
    def __init__(self, X, y, n_classifiers): 
        m, n = X.shape 
        self.n_classifiers = n_classifiers 
        self.X = X 
        self.y = y 
        self.mu = np.mean(X, axis=0) 
        self.sigma = np.std(X, axis=0) + 1e-30 
        self.training_weights = [np.ones((m,))] 
        self.models = None 
        self.alphas = None 

    def train(self): 
        self.models = [] 
        self.alphas = []
        X = self.X
        y = self.y  
        X_n = scale(X) 
        for n in range(self.n_classifiers): 
            weights = self.training_weights[-1]
            lr = LogisticRegression(C=1/LAMBDA, tol=1e-4, solver='sag', max_iter=EPOCH, verbose=0, n_jobs=-1) 
            lr.fit(X_n, y.flatten(), sample_weight=weights)  
            self.models.append(lr) 
            pred = lr.predict(X_n) 
            correct = pred == y 
            wrong = pred != y 
            epsilon = np.sum(weights[wrong]) / np.sum(weights) 
            print('epsilon:', epsilon)
            d = np.sqrt((1 - epsilon) / epsilon) 
            print('d:', d)
            alpha = np.log(d) 
            print('alpha', alpha)
            y_s = 2 * y - 1
            pred_s = 2 * pred - 1
            new_weights = weights * np.exp(-y_s * pred_s * alpha)             
            self.training_weights.append(new_weights) 
            self.alphas.append(alpha) 
        return 

    def predict(self, t): 
        results = [] 
        for lr in self.models: 
            t_n = (t - self.mu) / self.sigma 
            pred = lr.predict(t_n) 
            pred_s = 2 * pred - 1
            results.append(pred_s)  
        results = np.array(results) 
        results = np.average(results, axis=0, weights=self.alphas) 
        pred = (results > 0).astype(np.uint8) 
        return pred 

if __name__ == '__main__': 
    X_file = sys.argv[1] 
    y_file = sys.argv[2] 
    t_file = sys.argv[3] 
    pred_output_file = sys.argv[4] 
    X, y = read_training_data(X_file, y_file) 
    adaboost = Adaboost(X, y, n_classifiers=5) 
    adaboost.train() 
    with open('adaboost.pickle', 'wb') as f:
        pickle.dump(adaboost, f) 
        f.close() 
    t = read_testing_data(t_file) 
    pred = adaboost.predict(t) 
    write_prediction(pred_output_file, pred) 

