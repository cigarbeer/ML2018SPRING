import numpy as np 

from util import * 
import setting as st 

import sys 

if __name__ == '__main__':
    t_file = sys.argv[1]
    output_file = sys.argv[2]

    t = read_testing_data(t_file)
    theta = load_model(st.LOGISTIC_MODEL_PATH)
    mu, sigma = load_statistics(st.STATISTICS_PATH)
    t = normalize_testing_data(t, mu, sigma)
    t = add_bias_term(t)
    p = predict(theta, h, t)
    write_preditction(output_file, p)