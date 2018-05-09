import numpy as np 
import pandas as pd 
import sys 
from util import * 

if __name__ == '__main__':
    label_file_path = sys.argv[1] 
    unlabel_file_path = sys.argv[2]
    print(label_file_path, unlabel_file_path)