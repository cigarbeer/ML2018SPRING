# TRAINING_SET_DIR = '../dataset/train_classified' 
N_TRAINING_EXAMPLES = 9850 
N_CLASSES = 4251 
BATCH_SIZE = 256  
EPOCHS = 30 
N_XGB_TRAINING_EXAMPLES = BATCH_SIZE * 500
FEATURE_DIM = 1024   


MODEL_CHECKPOINT_PATH = './model.hdf5'
FEATURE_EXTRACTOR_PATH = './feature_extractor.hdf5' 
DTRAIN_PATH = './dtrain.pickle'
DLABEL_PATH = './dlabel.pickle' 
XGB_MODEL_PATH = './xgb.pickle'  

IMAGE_DATA_GENERATOR_TRAIN_KARGS = dict(
    featurewise_center=False, 
    samplewise_center=False, 
    featurewise_std_normalization=False, 
    samplewise_std_normalization=False, 
    zca_whitening=False, 
    zca_epsilon=1e-06, 
    rotation_range=30.0, 
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    shear_range=0.2, 
    zoom_range=0.2, 
    channel_shift_range=0.0, 
    fill_mode='nearest', 
    cval=0.0, 
    horizontal_flip=True, 
    vertical_flip=False, 
    rescale=1/255, 
    preprocessing_function=None, 
    data_format=None 
) 

IMAGE_DATA_GENERATOR_TEST_KARGS = dict(
    featurewise_center=False, 
    samplewise_center=False, 
    featurewise_std_normalization=False, 
    samplewise_std_normalization=False, 
    zca_whitening=False, 
    zca_epsilon=1e-06, 
    rotation_range=0.0, 
    width_shift_range=0.0, 
    height_shift_range=0.0, 
    shear_range=0.0, 
    zoom_range=0.0, 
    channel_shift_range=0.0, 
    fill_mode='nearest', 
    cval=0.0, 
    horizontal_flip=False, 
    vertical_flip=False, 
    rescale=1/255, 
    preprocessing_function=None, 
    data_format=None 
) 


FLOW_FROM_DIRECTORY_KARGS = dict(
    # directory=TRAINING_SET_DIR, 
    target_size=(224, 224), 
    color_mode='rgb', 
    classes=None, 
    class_mode='categorical', 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    seed=0, 
    save_to_dir=None, 
    save_prefix='', 
    save_format='png', 
    follow_links=False
)
