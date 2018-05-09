import numpy as np 
import pandas as pd 

def read_label_data(path):
    data = pd.read_table(path, header=None, names=['label', 'words'], sep='\+\+\+\$\+\+\+', engine='python') 
    data.words = data.words.str.strip().str.lower() 
    return data.label, data.words  

