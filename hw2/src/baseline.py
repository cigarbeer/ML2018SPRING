import numpy as np 
import pandas as pd 
import sys 


def read_training_data(X_file, y_file):
    X = pd.read_csv(X_file) 
    y = pd.read_csv(y_file, header=None) 
    return X.values, y.values

def normalize_data(X):
    mu = np.mean(X, axis=0) 
    sigma = np.std(X, axis=0) 
    X = (X - mu) / sigma 
    return X, mu, sigma 

def add_bias_term(X):
    return np.insert(X, obj=0, values=1, axis=1)

def initialize_theta(X, y):
    m, n = X.shape 
    return np.zeros((n, 1))

def sigmoid_clipped(z, epsilon=1e-8): 
    return np.clip(1/(1+np.exp(-z)), epsilon, 1-epsilon) 

def h(theta, X): 
    return sigmoid_clipped(np.dot(X, theta))

def J(theta, h, X, y, lmbda):
    m, n = X.shape 
    h_theta = h(theta, X)
    error_cost = -np.sum(y * np.log(h_theta) + (1 - y) * np.log(1 - h_theta))
    l2_cost = lmbda * (1 / 2) * np.dot(theta[1:].T, theta[1:])
    return (1 / m) * (error_cost + l2_cost) 

def J_gradient(theta, h, X, y, lmbda):
    m, n = X.shape 
    error_gradient = np.dot(X.T, h(theta, X) - y) 
    l2_cost_gradient = lmbda * theta 
    l2_cost_gradient[0] = 0 
    return (1 / m) * (error_gradient + l2_cost_gradient)

def gradient_descent(theta, h, J, X, y, lmbda, eta, n_epoch):
    m, n = X.shape 
    gradient_square_sum = np.zeros(theta.shape) 
    
    for epoch in range(n_epoch): 
        gradient = J_gradient(theta, h, X, y, lmbda) 
        gradient_square_sum = gradient**2 + gradient_square_sum 
        theta = theta - eta * gradient / np.sqrt(gradient_square_sum) 
        print(epoch, J(theta, h, X, y, lmbda)) 

    return theta 


def train(h, J, X, y, lmbda, eta, n_epoch):
    X = add_bias_term(X) 
    theta_init = initialize_theta(X, y)
    theta = gradient_descent(theta_init, h, J, X, y, lmbda, eta, n_epoch) 
    return theta 

def save_model(model_path, theta, mu, sigma):
    return np.save(model_path, np.array([theta, mu, sigma])) 
    
def load_model(model_path):
    return np.load(model_path)

def normalize_testing_data(t, mu, sigma):
    return (t - mu) / sigma 
    
def predict(theta, h, t):
    pred = h(theta, t) > 0.5 
    return pred.astype(np.int) 

def write_prediction(prediction_path, pred):
    df = pd.DataFrame(pred, columns=['label'])
    df.index = df.index + 1
    df.to_csv(prediction_path, index_label='id')
    return df 
