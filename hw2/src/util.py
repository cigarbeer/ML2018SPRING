import pandas as pd 
import numpy as np 

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

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def h(theta, X):
    return sigmoid(np.dot(X, theta))

def J(theta, h, X, y, lmbda=0):
    m, n = X.shape 
    h_theta = h(theta, X)
    l2_reg_cost = lmbda * (1 / 2) * np.dot(theta[1:].T, theta[1:])
    error_cost = -np.sum(y * np.log(h_theta) + (1 - y) * np.log(1 - h_theta))
    J = (1 / m) * (error_cost + l2_reg_cost)
    return J 

def gradient(theta, h, X, y, lmbda=0):
    m, n = X.shape
    l2_reg_grad = lmbda * theta 
    l2_reg_grad[0] = 0
    error_grad = np.dot(X.T, h(theta, X) - y)
    grad = (1 / m) * (error_grad + l2_reg_grad)
    return grad 

def gradient_descent(theta, h, X, y, eta=0.1, lmbda=0, n_iters=10000):
    m, n = X.shape
    theta_record = np.zeros((n_iters, *theta.shape))
    loss_record = np.zeros((n_iters, 1))

    theta_record[0] = theta 
    loss_record[0] = J(theta, h, X, y, lmbda)
    grad_square_sum = np.zeros(theta.shape)

    for i in range(1, n_iters):
        grad = gradient(theta, h, X, y, lmbda)
        grad_square_sum = grad**2 + grad_square_sum
        theta = theta - eta * grad / np.sqrt(grad_square_sum)

        theta_record[i] = theta 
        loss_record[i] = J(theta, h, X, y, lmbda)
        print('iteration:', i, 'loss:', loss_record[i])

    return theta, theta_record, loss_record

def save_model(m_path, theta):
    np.save(m_path, theta)
    return 

def save_statistics(s_path, mu, sigma):
    np.save(s_path, np.array([mu, sigma]))
    return 

def read_testing_data(t_file):
    t = pd.read_csv(t_file)
    return t.values

def normalize_testing_data(t, mu, sigma):
    return (t - mu) / sigma 

def predict(theta, h, t):
    pred = h(theta, t) > 0.5 
    return pred.astype(np.int8)

def write_preditction(p_file, pred):
    df = pd.DataFrame(pred, columns=['label'])
    df.index = df.index + 1
    df.to_csv(p_file, index_label='id')
    return df 

def get_shuffled_index(X, y):
    m, n = X.shape 
    sidx = np.random.permutation(m)
    return sidx 

def cross_validation(h, X, y, eta=0.1, lmbda=0, n_iters=10000, n_fold=1):
    m, n = X.shape 

    n_Val = np.int(m / n_fold)

    theta_init = initialize_theta(X, y)

    for fold in range(n_fold):
        sidx = get_shuffled_index(X, y)
        sX = X[sidx]
        sy = y[sidx]

        valX = sX[:n_Val] 
        valy = sy[:n_Val]
        trainX = sX[n_Val:]
        trainy = sy[n_Val:]

        theta, theta_record, loss_record = gradient_descent(theta_init, h, trainX, trainy, eta, lmbda, n_iters)

        val_error