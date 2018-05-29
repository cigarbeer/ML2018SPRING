import sys 
import os 

from feature_extraction import *

if __name__ == '__main__': 
    training_set_directory = sys.argv[1] 

    train_feature_extractor(training_set_dir) 

    