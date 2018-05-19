SEED = 0
WORDVECTOR_DIM = 256 
WORDVECTOR_WINDOW = 7 
WORDVECTOR_MIN_COUNT = 3
WORDVECTOR_N_ITER = 3

DENSE_DROPOUT_RATE = 0.2
RNN_DROPOUT_RATE = 0.3 

BATCH_SIZE = 128  
EPOCHS = 100  
VALIDATION_SPLIT = 0.1 

RNN_MODEL_CHECKPOINT_PATH = './rnn.hdf5' 
SEMISUPERVISED_RNN_MODEL_CHECKPOINT_PATH = './semirnn.hdf5' 
HW5_MODEL_PATH = './hw5.pickle' 

SEMISUPERVISED_THRESHOLD = 0.8

STOPWORDS = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
'by', 'for', 'with', 'about', 'against', 'between', 'into',
'through', 'during', 'before', 'after', 'above', 'below', 'to',
'from', 'up', 'down', 'in', 'out', 'on', 'off', 'under',
'again', 'further', 'then', 'once', 'here', 'there', 'when',
'where', 'all', 'both', 'each', 'few', 'more',
'most', 'other', 'some', 'own',
'same', 'so', 'than', 's', 't', 'now'])