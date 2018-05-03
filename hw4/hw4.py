import numpy as np 
import pandas as pd  
import sys 
import os 

from keras.layers import Input
from keras.layers import Dense
from keras.models import Model 

from sklearn.cluster import KMeans 

from keras.models import load_model 

IMAGE_SHAPE = (28, 28) 
IMAGE_FLATTEN_SHAPE = (28 * 28,) 

def read_data(file_name): 
    return np.load(file_name) 

def preprocess_data(X):
    m, *_ = X.shape 
    X = X / 255 
    X = X.reshape((m, -1)) 
    return X.astype(np.float32) 

def split_validation_set(X, ratio):
    m, n = X.shape  
    train_num = int((1 - ratio) * m) 
    return X[:train_num], X[train_num:] 

def build_autoencoder(input_shape): 
    input_layer = Input(shape=input_shape) 
    encode_layer_1 = Dense(units=128, activation='selu')(input_layer) 
    encode_layer_2 = Dense(units=64, activation='selu')(encode_layer_1) 
    
    latent_layer = Dense(units=32, activation='selu')(encode_layer_2) 

    decoder_layer_1 = Dense(units=64, activation='selu') (latent_layer) 
    decoder_layer_2 = Dense(units=128, activation='selu') (decoder_layer_1) 
    
    output_layer = Dense(units=input_shape[0], activation='sigmoid') (decoder_layer_2) 

    encoder = Model(inputs=input_layer, outputs=latent_layer) 
    autoencoder = Model(inputs=input_layer, outputs=output_layer) 
    encoder.compile(optimizer='nadam', loss='mse')
    autoencoder.compile(optimizer='nadam', loss='mse') 
    encoder.summary() 
    autoencoder.summary() 
    return encoder, autoencoder 

def train(autoencoder, X, epochs, batch_size): 
    X_train, X_validation = split_validation_set(X, ratio=0.07) 
    autoencoder.fit(
        x=X_train, 
        y=X_train, 
        epochs=epochs, 
        validation_data=(X_validation, X_validation), 
        batch_size=batch_size, 
        shuffle=True 
    ) 
    return 

def save_models(autoencoder, encoder, path): 
    autoencoder.save(os.path.join(path, 'autoencoder.h5')) 
    encoder.save(os.path.join(path, 'encoder.h5'))
    return 

def reduce_dimension(X, encoder): 
    X_reduced = encoder.predict(X) 
    m, n = X_reduced.shape 
    X_reduced = X_reduced.reshape((m, -1)) 
    return X_reduced 

def clustering(X, n_clusters): 
    kmeans = KMeans(
        n_clusters=n_clusters, 
        init='k-means++', 
        n_init=10, 
        max_iter=300, 
        tol=0.0001, 
        precompute_distances='auto', 
        verbose=1, 
        random_state=0, 
        copy_x=True, 
        n_jobs=-1, 
        algorithm='auto'
    ) 
    kmeans.fit(X) 
    return kmeans 

def read_testing_data(file_name): 
    t = pd.read_csv(file_name)  
    return t  

def predict(kmeans, t): 
    label1 = kmeans.labels_[t.image1_index] 
    label2 = kmeans.labels_[t.image2_index] 
    pred = (label1 == label2).astype(np.uint8) 
    return pred 

def save_predection(pred, file_name): 
    result = pd.DataFrame(columns=['Ans'], dtype=np.uint8) 
    result.Ans = pred 
    result.to_csv(file_name, index=True, index_label='ID') 
    return 



if __name__ == '__main__': 
    images_npy_path = sys.argv[1] 
    t_file = sys.argv[2] 
    pred_file_path = sys.argv[3] 
    encoder_path = sys.argv[4] 

    X = read_data(images_npy_path) 
    t = read_testing_data(t_file)
    encoder = load_model(encoder_path)   
     
    X_preprocessed = preprocess_data(X) 
    X_reduced = reduce_dimension(X_preprocessed, encoder) 
    kmeans = clustering(X_reduced, n_clusters=2) 
    
    pred = predict(kmeans, t) 
    save_predection(pred, pred_file_path)  
