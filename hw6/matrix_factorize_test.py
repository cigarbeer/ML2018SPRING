import sys 
import os 

from util import *
from setting import * 


if __name__ == '__main__': 
    test_data_path = sys.argv[1] 
    output_file_path = sys.argv[2] 
    movies_data_path = sys.argv[3] 
    users_data_path = sys.argv[4] 

    print('[read testing data]')
    tuid, tmid = read_testing_data(test_data_path) 
    print('[load matrix factorization model]')
    model = load_matrix_factorization_model() 
    print('[make prediction]')
    pred = predict(model, t=[tuid, tmid], batch_size=BATCH_SIZE) 
    print('[write prediction]')
    write_prediction(pred, output_file_path)  
    print('[done]')
