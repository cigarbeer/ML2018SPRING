import numpy as np 
import pandas as pd 
import sys 



if __name__ == '__main__':
    data_file_name = sys.argv[1]

    df = read_data(data_file_name)

    df = preprocess_data(df) 

    