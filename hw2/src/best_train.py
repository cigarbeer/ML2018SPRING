import numpy as np 
import pandas as pd 
from util import * 
import sys 
import best_setting  as st 



if __name__ == '__main__':
    X, y = read_training_data(st.TRAINGING_X_PATH, st.TRAINGING_Y_PATH)

    