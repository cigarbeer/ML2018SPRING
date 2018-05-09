import numpy as np 
import pandas as pd 
import logging 

import gensim 
from keras.preprocessing.text import Tokenizer 

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) 

STOPWORDS = set('for a an of the and to in on at or are am'.split(' '))

def read_label_data(path):
    data = pd.read_table(path, header=None, names=['label', 'words'], sep='\+\+\+\$\+\+\+', encoding='utf-8', engine='python') 
    return data.label, data.words.str.strip().str.lower() 

def read_unlabel_data(path):
    data = pd.read_table(path, header=None, names=['words'], encoding='utf-8', engine='python') 
    return data.words 