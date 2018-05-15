from keras.applications.inception_v3 import InceptionV3 
from keras.preprocessing import image 
from keras.models import Model 
from keras.layers import Dense 
from keras.layers import GlobalAveragePooling2D 
from keras.layers import Input 
from keras.layers import Dropout 

from keras.callbacks import EarlyStopping 
from keras.callbacks import ModelCheckpoint 

from keras import backend as K 

from final.util import settings as st 
from final.util import datagen as dg 

base_model = InceptionV3(input_tensor=Input(shape=(224, 224, 3)), weights='imagenet', include_top=False) 

x = base_model.output 
x = GlobalAveragePooling2D()(x) 
x = Dense(units=512, activation='selu')(x) 
predictions = Dense(units=4251, activation='softmax')(x) 

model = Model(inputs=base_model.input, outputs=predictions) 

for layer in base_model.layers:
    layer.trainable = False 

model.compile(optimizer='nadam', loss='categorical_crossentropy') 


model.fit_generator( 
    generator=dg.train_generator, 
    steps_per_epoch=int(st.N_TRAINING_EXAMPLES/st.BATCH_SIZE), 
    epochs=st.EPOCHS, 
    verbose=1, 
    callbacks=[
        ModelCheckpoint(st.MODEL_CHECKPOINT_PATH, monitor='val_acc', save_best_only=True, verbose=1), 
        EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=10, verbose=1)
    ], 
    validation_data=dg.test_generator, 
    validation_steps=int(st.N_TRAINING_EXAMPLES/st.BATCH_SIZE), 
    class_weight=None, 
    max_queue_size=10, 
    workers=1, 
    use_multiprocessing=True, 
    initial_epoch=0
) 

for layer in model.layers[:249]: 
    layer.trainable = False 
for layer in model.layers[249:]: 
    layer.trainable = True 

model.compile(optimizer='nadam', loss='categorical_crossentropy') 

model.fit_generator( 
    generator=dg.train_generator, 
    steps_per_epoch=int(st.N_TRAINING_EXAMPLES/st.BATCH_SIZE), 
    epochs=st.EPOCHS, 
    verbose=1, 
    callbacks=[
        ModelCheckpoint(st.MODEL_CHECKPOINT_PATH, monitor='val_acc', save_best_only=True, verbose=1), 
        EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=10, verbose=1)
    ], 
    validation_data=dg.test_generator, 
    validation_steps=int(st.N_TRAINING_EXAMPLES/st.BATCH_SIZE), 
    class_weight=None, 
    max_queue_size=10, 
    workers=1, 
    use_multiprocessing=True, 
    initial_epoch=0
) 