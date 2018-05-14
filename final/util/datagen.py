from keras.preprocessing.image import ImageDataGenerator 

from util import settings as st 

train_datagen = ImageDataGenerator(**st.IMAGE_DATA_GENERATOR_TRAIN_KARGS) 
test_datagen = ImageDataGenerator(**st.IMAGE_DATA_GENERATOR_TEST_KARGS) 

train_generator = train_datagen.flow_from_directory(**st.FLOW_FROM_DIRECTORY_KARGS) 
test_generator = test_datagen.flow_from_directory(**st.FLOW_FROM_DIRECTORY_KARGS) 