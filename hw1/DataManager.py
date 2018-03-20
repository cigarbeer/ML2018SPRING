import numpy as np 
import pandas as pd 

class DataManager:
    def __init__(self):
        self.feature = None
        self.rdf = None # raw df
        self.data = None # dictionary of processed dfs data
        self.X = None
        self.y = None
        self.tdata = None
        
    def read_training_data(self, file_name):
        df = pd.read_csv(file_name, encoding='big5', na_values=['NR'])
        df = df.fillna(0.0)
        df = df.rename(columns={
            df.columns[0]: 'date',
            df.columns[1]: 'location',
            df.columns[2]: 'feature'
        })
        df.date = pd.to_datetime(df.date)
        df = df.drop(columns=['location'])
        self.feature = sorted(df.feature.unique())
        self.rdf = df
        self.rdf_to_data()
        return 
    
    def preprocess_mdfs(self, mdfs):
        for m, df in mdfs.items():
            cols = df.columns
            for col in cols:
                df = df[df[col] > 0.0]
            
            mdfs[m] = df
            
        return mdfs
    
    def rdf_to_data(self):
        rdf = self.rdf
        data = {month: {feature: df.drop(columns=['date', 'feature']).values.flatten() for feature, df in mdf.groupby(mdf.feature)} for month, mdf in rdf.groupby(rdf.date.dt.month)}
        self.data = data
        return 
    
    def select_feature_to_mdfs(self, feature_list=None):
        if feature_list is None:
            feature_list = self.feature

        mdfs = {month: pd.DataFrame(columns=sorted(feature_list)) for month in range(1, 13)}

        for month, fdata in self.data.items():
            for feature in feature_list:
                mdfs[month][feature] = fdata[feature]
            
        return mdfs
    
    
    def chunk_examples(self, mdfs, chunk_size):
        X = []
        y = []
        for month, mdf in mdfs.items():
            nrows, ncols = mdf.shape

            for i in range(nrows-chunk_size):
                X.append(mdf.iloc[i:i+chunk_size].values.flatten())
                y.append(mdf['PM2.5'].iloc[i+chunk_size])
                
        self.X = np.array(X)
        self.y = np.array(y).reshape((-1, 1))
        return self.X, self.y
    
    def get_shuffle_index(self):
        m, n = self.X.shape
        idx = np.random.permutation(np.arange(m))
        return idx
    
    def read_testing_data(self, file_name):
        df = pd.read_csv(file_name, header=None, na_values=['NR'])
        df = df.rename(columns={0: 'id', 1: 'feature'})
        df = df.fillna(0.0)
        self.tdata = df
        return
    
    def select_testing_feature(self, feature_list=None):
        if feature_list is None:
            feature_list = self.feature
            
        iddfs = {i: df for i, df in self.tdata.groupby(self.tdata.id)}
        
        for i, df in iddfs.items():
            columns = df.feature
            df = df.drop(columns=['id', 'feature']).T
            df.columns = columns
            df = df[sorted(feature_list)]
            iddfs[i] = df
        return iddfs

