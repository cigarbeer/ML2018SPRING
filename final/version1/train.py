import sys 
import os 

import settings as st 

from feature_extraction import *
from classfication import * 

if __name__ == '__main__': 
    training_set_directory = sys.argv[1] 

    print('[get data generator]')
    train_datagen = ImageDataGenerator(**st.IMAGE_DATA_GENERATOR_TRAIN_KARGS) 
    test_datagen = ImageDataGenerator(**st.IMAGE_DATA_GENERATOR_TEST_KARGS) 

    train_generator = train_datagen.flow_from_directory(directory=training_set_directory, **st.FLOW_FROM_DIRECTORY_KARGS) 
    test_generator = test_datagen.flow_from_directory(directory=training_set_directory, **st.FLOW_FROM_DIRECTORY_KARGS) 

    print('[train feature extractor]')
    train_feature_extractor(train_generator, test_generator) 

    print('[load feature extractor]')
    feature_extractor = load_feature_extractor()

    print('[get xgb dmatrix]')
    dtrain, dlabel = get_dmatrix(feature_extractor, train_generator) 

    print('[train xgb classifier]')
    train_xgb_classifier(dtrain, dlabel) 