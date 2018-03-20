import numpy as np 
import pandas as pd 
import sys 

from DataManager import DataManager 


def normalize_data(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_n = (X - mu) / sigma
    return X_n, mu, sigma

def norm_eq(X, y):
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)
    return theta

def predict(theta, instance):
    h_theta = np.dot(instance, theta)
    return h_theta

def h(X, theta):
    return np.dot(X, theta)

def J(theta, h, X, y):
    m, n = X.shape
    square_error = np.sum((h(X, theta) - y)**2)
    return (1 / 2) * (1 / m) * square_error

def gradient_descent(theta, h, X, y, eta, n_iters):
    m, n = X.shape
    theta_record = np.zeros((n_iters, *theta.shape))
    loss_record = np.zeros((n_iters, 1))
    
    theta_record[0] = theta
    loss_record[0] = J(theta, h, X, y)
    grad_square_sum = np.zeros(theta.shape)
    for i in range(1, n_iters):
        grad = (1 / m) * np.dot(X.T, (h(X, theta) - y))
        
        grad_square_sum = grad**2 + grad_square_sum
        
        theta = theta - eta * grad / np.sqrt(grad_square_sum)
        theta_record[i] = theta
        loss_record[i] = J(theta, h, X, y)
        
    return theta, theta_record, loss_record

def initialize_theta(X):
    m, n = X.shape
    
    return np.random.normal(loc=0, scale=1, size=(n+1, 1))

def find_example_error(theta, h, J, X, y):
    error_list = []

    for i, example in enumerate(zip(X, y)):
        xx, yy = example
        xx = xx.reshape((1, -1))
        yy = yy.reshape((1, 1))
        error_list.append((J(theta, h, xx, yy), i, xx, yy))

    return error_list

def validate_data(rate, h, J, X, y):
    norm_theta = norm_eq(X, y)
    error_list = sorted(find_example_error(norm_theta, h, J, X, y), reverse=True)
    to_drop = error_list[0:int(rate*len(error_list))]
    to_drop_idx = [idx for err, idx, xx, yy in to_drop]
    return np.delete(X, to_drop_idx, axis=0), np.delete(y, to_drop_idx, axis=0)


SELECTED_FEATURE = ['PM2.5', 'PM10']
CHUNK_SIZE = 5
VALIDATE_RATE = 0.05
ETA = 0.1
N_ITERS = 200000

if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    dm = DataManager()

    dm.read_training_data(input_file)

    mdfs = dm.select_feature_to_mdfs(SELECTED_FEATURE)

    mdfs = dm.preprocess_mdfs(mdfs)

    X, y = dm.chunk_examples(mdfs, chunk_size=CHUNK_SIZE)

    X, y = validate_data(VALIDATE_RATE, h, J, X, y)

    X_n, mu, sigma = normalize_data(X)

    init_theta = initialize_theta(X_n)

    theta, theta_record, loss_record = gradient_descent(init_theta, h, np.insert(X_n, obj=0, values=1, axis=1), y, eta=ETA, n_iters=N_ITERS)

    np.save('model_c%d_v%d'%(CHUNK_SIZE, int(VALIDATE_RATE*100)), )