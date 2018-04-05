import numpy as np 
import pandas as pd 
from util import * 
import sys 
import generative_setting as st 



if __name__ == '__main__':
    X_file = sys.argv[1]
    y_file = sys.argv[2]

    X, y = read_training_data(X_file, y_file) 

    mu0, mu1, cov, m0, m1 = get_generative_model(X, y)

    save_gaussian_model(st.GAUSSIAN_MODEL_PATH, mu0, mu1, cov, m0, m1)

    

