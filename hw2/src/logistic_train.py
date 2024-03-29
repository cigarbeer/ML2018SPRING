import numpy as np 
import pandas as pd 
from util import * 
import sys 
import logistic_setting as st 

if __name__ == '__main__':
    X_file = sys.argv[1] 
    y_file = sys.argv[2]
    X, y = read_training_data(X_file, y_file)
    X, mu, sigma = normalize_data(X)
    theta, loss_record = train(h, J, X, y, eta=st.ETA, lmbda=st.LAMBDA, epoch=st.EPOCH, n_minibatch=st.N_MINIBATCH)
    save_model(st.LOGISTIC_MODEL_PATH, theta)
    save_statistics(st.STATISTICS_PATH, mu, sigma)
    
