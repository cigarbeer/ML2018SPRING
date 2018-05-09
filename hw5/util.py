import numpy as np 
import pandas as pd 

def read_label_data(path):
    data = pd.read_table(path, sep=' +++$+++ ')
    return data 

