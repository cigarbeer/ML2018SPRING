import numpy as np 
import pandas as pd 
from util import * 
import sys 
import best_setting  as st 

from sklearn.linear_model import LogisticRegressionCV 
from sklearn.covariance import EllipticEnvelope 
from sklearn.preprocessing import scale 
from sklearn.svm import SVC
from sklearn.feature_selection import RFE 


if __name__ == '__main__':
    X, y = read_training_data(st.TRAINGING_X_PATH, st.TRAINGING_Y_PATH)

    X = scale(X) 

    ee = EllipticEnvelope()
    ee.fit(X, y.flatten())

    inliers_mask = ee.predict(X)
    

