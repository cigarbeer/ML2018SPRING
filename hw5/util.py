import settings as st 

import logging 
import pickle 
from string import punctuation 
import multiprocessing 

import numpy as np 
np.random.seed(st.SEED) 
import pandas as pd 

import gensim 
from gensim.corpora import Dictionary 

from keras.utils import to_categorical 
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
 

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) 

def read_label_data(path):
    data = pd.read_table(path, header=None, names=['label', 'texts'], sep='\+\+\+\$\+\+\+', encoding='utf-8', engine='python') 
    return data.label, data.texts.str.strip().str.lower() 

def read_unlabel_data(path):
    data = pd.read_table(path, header=None, names=['texts'], sep='\r\n', encoding='utf-8', engine='python') 
    return data.texts.str.strip().str.lower() 

def read_testing_data(path): 
    data = pd.read_table(path, header=0, names=['line'], encoding='utf-8', engine='python') 
    data = data.line.str.split(pat=',', n=1, expand=True) 
    data = data.rename(columns={0: 'id', 1: 'text'})
    return data.id, data.text.str.strip().str.lower() 

def concat_data(label_texts, unlabel_texts): 
    return pd.concat([label_texts, unlabel_texts]) 

def texts2corpus(texts): 
    def split_texts(texts):
        return texts.apply(text_to_word_sequence, lower=True, split=' ')
    return split_texts(texts) 


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
        # compute_loss=True, 
        seed=0, 
        workers=multiprocessing.cpu_count()
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
        dropout=st.RNN_DROPOUT_RATE, 
        recurrent_dropout=st.RNN_DROPOUT_RATE,
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
    model.add(Dense(units=256, activation='selu')) 
    model.add(Dropout(rate=st.DENSE_DROPOUT_RATE))
    model.add(Dense(units=256, activation='selu'))
    model.add(Dropout(rate=st.DENSE_DROPOUT_RATE))

    model.add(Dense(units=2, activation='softmax')) 
    model.summary() 

    model.compile(
        optimizer='nadam', 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    ) 
    return model 


def train(model, X, y, batch_size, epochs, validation_split=0.0, save_model_path=None, validation_data=None): 
    # (train_X, train_y), (val_X, val_y) = split_validation_set(X, y, rate=0.1) 
    callbacks = [
        EarlyStopping(
            monitor='val_acc', 
            min_delta=1e-4, 
            patience=10, 
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
        y=to_categorical(y), 
        batch_size=batch_size, 
        epochs=epochs, 
        verbose=1, 
        callbacks=callbacks, 
        validation_split=validation_split, 
        validation_data=validation_data, 
        shuffle=True, 
        # class_weight=None, 
        # sample_weight=None, 
        initial_epoch=0
        # steps_per_epoch=None, 
        # validation_steps=None
    ) 
    return model 

def predict(model, t, batch_size): 
    prob = model.predict(x=t, batch_size=batch_size, verbose=1) 
    pred = np.argmax(prob, axis=1) 
    return pred 

def write_prediction(pred, path): 
    df = pd.DataFrame(columns=['label'], data=pred) 
    df.to_csv(path, index=True, index_label='id') 
    return 

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

def get_semisupervised_data(model, uidseqpad, threshold): 
    prob = model.predict(x=uidseqpad, batch_size=st.BATCH_SIZE, verbose=1) 
    # maxprob = np.max(prob, axis=1) 
    positiveidx = np.where(prob[:, 1] > threshold)[0] 
    negativeidx = np.where(prob[:, 0] > threshold)[0] 
    positivelabel = np.ones(positiveidx.shape) 
    negativelabel = np.zeros(negativeidx.shape) 
    semilabel = np.concatenate((positivelabel, negativelabel)) 
    semidata = np.concatenate((uidseqpad[positiveidx], uidseqpad[negativeidx]))
    return semilabel, semidata

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
    stop_ids = [dct.token2id.get(stopword) for stopword in st.STOPWORDS if not dct.token2id.get(stopword) is None]  
    dct.filter_tokens(bad_ids=stop_ids, good_ids=None) 
    dct.compactify() 
    return dct 

def split_validation_set(X, y, rate): 
    m, n = X.shape
    n_train = int(rate * m) 
    train_X = X[:n_train] 
    train_y = y[:n_train] 
    val_X = X[n_train:] 
    val_y = y[n_train:] 
    return (train_X, train_y), (val_X, to_categorical(val_y)) 

class Hw5:
    def __init__(self): 
        self.max_document_size = None 
        self.tokenizer = None 
        self.wordvector = None 
    
    def get_wordvector(self, corpus=None): 
        if corpus is None: 
            return self.wordvector  
        if self.wordvector is None: 
            self.wordvector = word2vec(corpus=corpus, dim=st.WORDVECTOR_DIM, window=st.WORDVECTOR_WINDOW, min_count=st.WORDVECTOR_MIN_COUNT, n_iter=st.WORDVECTOR_N_ITER)
        return self.wordvector 

    def get_tokenizer(self, texts=None):
        if texts is None: 
            return self.tokenizer 
        if self.tokenizer is None: 
            self.tokenizer = build_tokenizer(texts) 
        return self.tokenizer 

    def load_rnn_model(self): 
        return load_model(st.RNN_MODEL_CHECKPOINT_PATH) 

    def load_semisupervised_rnn_model(self): 
        return load_model(st.SEMISUPERVISED_RNN_MODEL_CHECKPOINT_PATH) 

