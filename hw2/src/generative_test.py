import numpy as np 
import pandas as pd 
from util import * 
import sys 
import generative_setting as st 


if __name__ == '__main__':
    t_file = sys.argv[1]
    output_file = sys.argv[2]  

    t = read_testing_data(t_file)

    mu0, mu1, cov, m0, m1 = load_gaussian_model(st.GAUSSIAN_MODEL_PATH)

    p = gen_predict(t, mu0, mu1, cov, m0, m1) 

    write_preditction(output_file, p)

    
