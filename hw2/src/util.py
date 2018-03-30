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
    return np.nan_to_num(X), mu, sigma 

def add_bias_term(X):
    return np.insert(X, obj=0, values=1, axis=1)

def initialize_theta(X, y):
    m, n = X.shape
    return np.zeros((n, 1))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def h(theta, X):
    return sigmoid(np.dot(X, theta))

def J(theta, h, X, y, lmbda):
    m, n = X.shape 
    h_theta = h(theta, X)
    l2_reg_cost = lmbda * (1 / 2) * np.dot(theta[1:].T, theta[1:])
    error_cost = -np.sum(y * np.nan_to_num(np.log(h_theta)) + (1 - y) * np.nan_to_num(np.log(1 - h_theta)))
    J = (1 / m) * (error_cost + l2_reg_cost)
    return J 

def gradient(theta, h, X, y, lmbda):
    m, n = X.shape
    l2_reg_grad = lmbda * theta 
    l2_reg_grad[0] = 0
    error_grad = np.dot(X.T, h(theta, X) - y)
    grad = (1 / m) * (error_grad + l2_reg_grad)
    return grad 

def gradient_descent(theta, h, J, X, y, eta, lmbda, epoch, n_minibatch):
    m, n = X.shape
    loss_record = []

    grad_square_sum = np.zeros(theta.shape)

    for ep in range(epoch):
        sidx = get_shuffled_index(X, y)
        mbX = np.array_split(X, indices_or_sections=n_minibatch, axis=0)
        mby = np.array_split(y, indices_or_sections=n_minibatch, axis=0)
        for i, (mX, my) in enumerate(zip(mbX, mby)):
            grad = gradient(theta, h, mX, my, lmbda)
            grad_square_sum = grad**2 + grad_square_sum
            adagrad = np.nan_to_num(grad / np.sqrt(grad_square_sum))
            theta = theta - eta * adagrad

            loss_record.append(J(theta, h, mX, my, lmbda))
            print('epoch:', ep, 'mini batch:', i , 'loss:', loss_record[-1])

    return theta, loss_record

def train(h, J, X, y, eta, lmbda, epoch, n_minibatch):
    X = add_bias_term(X) 
    theta_init = initialize_theta(X, y)
    theta, loss_record = gradient_descent(theta_init, h, J, X, y, eta, lmbda, epoch, n_minibatch)
    return theta, loss_record


def test(theta, h, mu, sigma, tX, ty):
    tX = normalize_testing_data(tX, mu, sigma) 
    tX = add_bias_term(tX)
    return J(theta, h, tX, ty, lmbda=0)


def save_model(m_path, theta):
    np.save(m_path, theta)
    return 

def save_statistics(s_path, mu, sigma):
    np.save(s_path, np.array([mu, sigma]))
    return 

def load_model(m_path):
    return np.load(m_path)

def load_statistics(s_path):
    return np.load(s_path)

def read_testing_data(t_file):
    t = pd.read_csv(t_file)
    return t.values

def normalize_testing_data(t, mu, sigma):
    t = (t - mu) / sigma 
    return np.nan_to_num(t)

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

