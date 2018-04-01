import numpy as np 
import pandas as pd 
from util import * 
import sys 
import generative_setting as st 



if __name__ == '__main__':
    X, y = read_training_data(st.TRAINGING_X_PATH, st.TRAINGING_Y_PATH) 

    mu0, mu1, cov, m0, m1 = get_generative_model(X, y)

    save_gaussian_model(st.GAUSSIAN_MODEL_PATH, mu0, mu1, cov, m0, m1)

    

