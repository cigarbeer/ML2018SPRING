import logging 
import pickle 
from string import punctuation 

import numpy as np 
import pandas as pd 

import gensim 
from gensim.corpora import Dictionary 

from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.text import text_to_word_sequence 
from keras.preprocessing.sequence import pad_sequences 
from keras.models import Sequential 
from keras.models import save_model 
from keras.models import load_model 
from keras.layers import Embedding 
from keras.layers import GRU 
from keras.layers import Dense 
from keras.layers import Dropout 
from keras.callbacks import EarlyStopping 
from keras.callbacks import ModelCheckpoint 

from settings import STOPWORDS 

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) 

def read_label_data(path):
    data = pd.read_table(path, header=None, names=['label', 'texts'], sep='\+\+\+\$\+\+\+', encoding='utf-8', engine='python') 
    return data.label, data.texts.str.strip().str.lower() 

def read_unlabel_data(path):
    data = pd.read_table(path, header=None, names=['texts'], sep='\r\n', encoding='utf-8', engine='python') 
    return data.texts.str.strip().str.lower() 

def read_testing_data(path): 
    data = pd.read_csv(path, encoding='utf-8', engine='python')
    return data.id, data.text.str.strip().lower() 

def concat_data(label_texts, unlabel_texts): 
    return pd.concat([label_texts, unlabel_texts]) 


def build_tokenizer(texts, num_words=None): 
    tkn = Tokenizer(
        num_words=num_words, 
        # filters=punctuation, 
        lower=True, 
        split=' ', 
        char_level=False 
    ) 
    tkn.fit_on_texts(texts) 
    return tkn 

def texts2idseq(texts, tokenizer): 
    return tokenizer.texts_to_sequences(texts) 

def pad_idseq(idseq, max_len=None): 
    return pad_sequences(idseq, maxlen=max_len, padding='post', truncating='post') 

def convert_embedding_weights(tokenizer, wordvector): 
    vocab_size = len(tokenizer.word_index) 
    vector_dim = wordvector.vector_size 
    weights = np.zeros((vocab_size, vector_dim)) 
    for word, idx in tokenizer.word_index.items(): 
        if word in wordvector: 
            weights[idx] = wordvector[word] 
    return weights 

def word2vec(corpus, dim, window, min_count, n_iter): 
    return gensim.models.Word2Vec(
        sentences=corpus, 
        sg=1,
        size=dim, 
        window=window, 
        min_count=min_count, 
        iter=n_iter, 
        compute_loss=True, 
        seed=0
    ) 

def wordvector_rnn_classifier(wordvector, tokenizer, max_document_size):
    model = Sequential() 
    model.add(Embedding(
        input_dim=len(tokenizer.word_index), 
        output_dim=wordvector.vector_size, 
        input_length=max_document_size, 
        weights=[convert_embedding_weights(tokenizer, wordvector)], 
        trainable=False 
    ))
    model.add(GRU(
        units=wordvector.vector_size, 
        activation='tanh', 
        recurrent_activation='hard_sigmoid', 
        use_bias=True, 
        dropout=0.0, 
        recurrent_dropout=0.0,
        kernel_initializer='glorot_uniform', 
        recurrent_initializer='orthogonal', 
        bias_initializer='zeros', 
        kernel_regularizer=None, 
        recurrent_regularizer=None,
        bias_regularizer=None, 
        activity_regularizer=None, 
        kernel_constraint=None, 
        recurrent_constraint=None, 
        bias_constraint=None
    ))
    model.add(Dense(units=128, activation='selu')) 
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=128, activation='selu'))
    model.add(Dropout(rate=0.2))

    model.add(Dense(units=1, activation='sigmoid')) 
    model.summary() 

    model.compile(
        optimizer='nadam', 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    ) 
    return model 


def train(model, X, y, batch_size, epochs, validation_split, save_model_path): 
    # (train_X, train_y), (val_X, val_y) = split_validation_set(X, y, rate=0.1) 
    callbacks = [
        EarlyStopping(
            monitor='val_acc', 
            min_delta=1e-4, 
            patience=30, 
            verbose=1
        ), 
        ModelCheckpoint(
            filepath=save_model_path,
            monitor='val_acc', 
            save_best_only=True, 
            save_weights_only=False,
            verbose=1
        )
    ] 
    model.fit(
        x=X, 
        y=y, 
        batch_size=batch_size, 
        epochs=epochs, 
        verbose=1, 
        callbacks=callbacks, 
        validation_split=validation_split, 
        validation_data=None, 
        shuffle=True, 
        class_weight=None, 
        sample_weight=None, 
        initial_epoch=0, 
        steps_per_epoch=None, 
        validation_steps=None
    )
    return model 

def save_object(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f) 
        f.close()
    return 

def load_object(path): 
    with open(path, 'rb') as f: 
        obj = pickle.load(f) 
        f.close()
    return obj 


def texts2bow(texts, dct): 
    corpus = texts2corpus(texts) 
    bow = corpus2bow(corpus, dct) 
    return bow 

def corpus2bow(corpus, dct): 
    bow = [dct.doc2bow(doc) for doc in corpus] 
    return bow 

def bow2dense(bow, dct): 
    dense = gensim.matutils.corpus2dense(bow, num_terms=len(dct)) 
    return dense.T   

def build_dictionary(corpus, min_count, max_dct_size): 
    dct = Dictionary(corpus) 
    dct.filter_extremes(
        no_below=min_count, 
        no_above=0.5, 
        keep_n=max_dct_size, 
        keep_tokens=None 
    ) 
    # dct.filter_n_most_frequent(
    #     remove_n=3
    # )
    stop_ids = [dct.token2id.get(stopword) for stopword in STOPWORDS if not dct.token2id.get(stopword) is None]  
    dct.filter_tokens(bad_ids=stop_ids, good_ids=None) 
    dct.compactify() 
    return dct 

def texts2corpus(texts): 
    def split_texts(texts):
        return texts.apply(text_to_word_sequence, lower=True, split=' ')
    return split_texts(texts) 

def split_validation_set(X, y, rate): 
    m ,n = X.shape
    n_train = int(rate * m) 
    train_X = X[:n_train] 
    train_y = y[:n_train] 
    val_X = X[n_train:] 
    val_y = y[n_train:] 
    return (train_X, train_y), (val_X, val_y) 

class Hw5:
    def __init__(self): 
        self.max_document_size = None 
        self.tokenizer = None 
        self.wordvector = None 
        self.rnn_model = None 
        self.bow_model = None 