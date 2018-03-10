import numpy as np 
import pandas as pd 
import sys 

feature_name = [
    'AMB_TEMP',
    'CH4',
    'CO',
    'NMHC',
    'NO',
    'NO2',
    'NOx',
    'O3',
    'PM10',
    'PM2.5',
    'RAINFALL',
    'RH',
    'SO2',
    'THC',
    'WD_HR',
    'WIND_DIREC',
    'WIND_SPEED',
    'WS_HR'
]

def read_data(data_file_name, encoding='big5'):
    df = pd.read_csv(data_file_name, encoding=encoding, na_values=['NR'])
    df = df.fillna(0)
    return df

def preprocess_data(df):
    df = df.rename(
        columns={
            df.columns[0]: 'date',
            df.columns[1]: 'location',
            df.columns[2]: 'feature'
    })
    
    
    return df



if __name__ == '__main__':
    data_file_name = sys.argv[1]

    df = read_data(data_file_name)

    df = preprocess_data(df) 

    