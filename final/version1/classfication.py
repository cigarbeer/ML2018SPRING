import pickle 

import xgboost as xgb 
import numpy as np 

import settings as st 

def save_object(obj, path): 
    with open(path, mode='wb') as f: 
        pickle.dump(obj, f)
        f.close() 
    return 

def get_dmatrix(feature_extractor, generator): 
    dtrain = np.zeros((st.N_XGB_TRAINING_EXAMPLES, st.FEATURE_DIM)) 
    dlabel = np.zeros((st.N_XGB_TRAINING_EXAMPLES,)) 
    # dlabel = np.zeros((st.N_XGB_TRAINING_EXAMPLES, st.N_CLASSES)) 
    start = 0
    for X, y in generator: 
        batch_size, *_ = X.shape 
        if not (batch_size == st.BATCH_SIZE): 
            continue 
        print('start position', start) 
        X = feature_extractor.predict(X, batch_size=st.BATCH_SIZE, verbose=1)  
        y = np.argmax(y, axis=1) 
        print('X shape', X.shape) 
        print('y shape', y.shape) 
        dtrain[start:start+st.BATCH_SIZE] = X 
        dlabel[start:start+st.BATCH_SIZE] = y 
        start = start + st.BATCH_SIZE 
        if start == st.N_XGB_TRAINING_EXAMPLES:
            break 
    save_object(dtrain, st.DTRAIN_PATH) 
    save_object(dlabel, st.DLABEL_PATH) 
    # return xgb.DMatrix(data=dtrain, label=dlabel)
    return dtrain, dlabel  

def train_xgb_classifier(dtrain, dlabel): 
    xgbc = xgb.XGBClassifier(objective='multi:softmax') 
    xgbc.fit(dtrain, dlabel) 
    save_object(xgbc, st.XGB_MODEL_PATH) 
    return xgbc 

def predict(feature_extractor, xgb_classifier, generator): 
    n_testing_examples = len(generator.filenames) 
    t = np.zeros((n_testing_examples, st.FEATURE_DIM)) 
    for X, y in generator: 
         X = feature_extractor.predict(X) 

    pred = model.predict(t)