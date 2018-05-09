import numpy as np 
import pandas as pd 

def read_label_data(path):
    data = pd.read_table(path, header=None, names=['label', 'words'], sep='\+\+\+\$\+\+\+')
    return data.label, data.words  

