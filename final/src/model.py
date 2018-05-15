from keras.models import Model 

from keras.layers import Input 
from keras.layers import Dense
from keras.layers import Conv2D 
from keras.layers import Dropout 
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D 
from keras.layers import GlobalAveragePooling2D 

from keras.optimizers import Nadam 

from keras.callbacks import EarlyStopping 
from keras.callbacks import ModelCheckpoint 

from keras.applications.vgg16 import VGG16 

input_layer = Input(shape=(224, 224, 3), name='input_layer') 
vgg16 = VGG16(input_tensor=input_layer, weights='imagenet', include_top=False)(input_layer) 
global_average_pooling_layer = GlobalAveragePooling2D(name='global_average_pooling_layer')(vgg16.output) 

dense_layer_1 = Dense(units=256, activation='selu')(global_average_pooling_layer) 
batch_normalization_layer_1 = BatchNormalization()(dense_layer_1) 
dropout_layer_1 = Dropout(rate=0.25)(batch_normalization_layer_1)

dense_layer_2 = Dense(units=256, activation='selu')(dropout_layer_1) 
batch_normalization_layer_2 = BatchNormalization()(dense_layer_2) 
dropout_layer_2 = Dropout(rate=0.25)(batch_normalization_layer_2)
output_layer = Dense(units=4251, activation='softmax', name='output_layer')(dropout_layer_2) 
model = Model(inputs=input_layer, outputs=output_layer)

# for layer in vgg16.layers: 
#     layer.trainable = False 

# model.compile(
#     optimizer=Nadam(lr=0.001), 
#     loss='categorical_crossentropy'
# ) 
