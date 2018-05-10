import numpy as np 
import pandas as pd 
import logging 
import pickle 
from string import punctuation 
import gensim 
from gensim.corpora import Dictionary
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.text import text_to_word_sequence 
from keras.models import Sequential 
from keras.models import save_model 
from keras.models import load_model 

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) 

STOPWORDS = set('for a an of the and to in on at or are i am was 2'.split(' '))
MAX_DICTIONATY_SIZE = 10000 


def read_label_data(path):
    data = pd.read_table(path, header=None, names=['label', 'texts'], sep='\+\+\+\$\+\+\+', encoding='utf-8', engine='python') 
    return data.label, data.texts.str.strip().str.lower() 

def read_unlabel_data(path):
    data = pd.read_table(path, header=None, names=['texts'], sep='\r\n', encoding='utf-8', engine='python') 
    return data.texts.str.strip().str.lower()

def concat_data(label_texts, unlabel_texts): 
    return pd.concat([label_texts, unlabel_texts]) 

def texts2corpus(texts): 
    return split_texts(texts) 

def build_dictionary(corpus): 
    dct = Dictionary(corpus) 
    dct.filter_extremes(
        no_below=3, 
        no_above=0.5, 
        keep_n=10000, 
        keep_tokens=None 
    ) 
    # dct.filter_n_most_frequent(
    #     remove_n=3
    # )
    stop_ids = [dct.token2id.get(stopword) for stopword in STOPWORDS if not dct.token2id.get(stopword) is None]  
    dct.filter_tokens(bad_ids=stop_ids, good_ids=None) 
    dct.compactify() 
    return dct 

def tokenize(words, num_words): 
    tkn = Tokenizer(
        num_words=num_words, 
        filters=punctuation, 
        lower=True, 
        split=' ', 
        char_level=False 
    ) 
    tkn.fit_on_texts(words) 
    return tkn 


def split_texts(texts):
    return texts.apply(text_to_word_sequence, filters=punctuation, lower=True, split=' ')

def word2vec(words, dim, window, min_count, n_iter): 
    return gensim.models.Word2Vec(
        sentences=words, 
        sg=1,
        size=dim, 
        window=window, 
        min_count=min_count, 
        iter=n_iter, 
        compute_loss=True, 
        seed=0
    ) 


def save_object(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f) 
        f.close()
    return 

def load_object(path): 
    with open(path, 'rb') as f: 
        obj = pickle.load(f) 
        f.close()
    return obj 


