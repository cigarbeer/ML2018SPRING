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
    dlabel = np.zeros((st.N_XGB_TRAINING_EXAMPLES, st.N_CLASSES)) 
    start = 0
    for X, y in generator: 
        X = feature_extractor.predict(X) 
        dtrain[start:start+st.BATCH_SIZE] = X 
        dlabel[start:start+st.BATCH_SIZE] = y 
        start = start + st.BATCH_SIZE 
        if start == st.N_XGB_TRAINING_EXAMPLES:
            break 

    # return xgb.DMatrix(data=dtrain, label=dlabel)
    return dtrain, dlabel  

def train_xgb_classifier(dtrain, dlabel): 
    xgbc = xgb.XGBClassifier(objective='multi:softmax') 
    xgbc.fit(dtrain, dlabel) 
    save_object(xgbc, st.XGB_MODEL_PATH) 
    return xgbc 