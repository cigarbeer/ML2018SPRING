import numpy as np 
import pandas as pd 
import logging 

import gensim 
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.text import text_to_word_sequence 
from keras.models import Sequential 
from keras.models import save_model 
from keras.models import load_model 

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) 

STOPWORDS = set('for a an of the and to in on at or are am'.split(' '))

def read_label_data(path):
    data = pd.read_table(path, header=None, names=['label', 'words'], sep='\+\+\+\$\+\+\+', encoding='utf-8', engine='python') 
    return data.label, data.words.str.strip().str.lower() 

def read_unlabel_data(path):
    data = pd.read_table(path, header=None, names=['words'], sep='\r\n', encoding='utf-8', engine='python') 
    return data.words.str.strip().str.lower()

def concat_data(label_data, unlabel_data): 
    return pd.concat([label_data, unlabel_data]) 

def split_words(words):
    return words.apply(text_to_word_sequence)

def word2vec(words, dim, window, min_count, n_iter): 
    return gensim.models.Word2Vec(sentences=words, size=dim, window=window, min_count=min_count, iter=n_iter, compute_loss=True, seed=0) 


