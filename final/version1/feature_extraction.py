from keras.applications.inception_v3 import InceptionV3 

from keras.preprocessing import image 

from keras.models import Model 

from keras.layers import Dense 
from keras.layers import GlobalAveragePooling2D 
from keras.layers import Input 

from keras.callbacks import EarlyStopping 
from keras.callbacks import ModelCheckpoint 
from keras.callbacks import ReduceLROnPlateau 

from . import settings as st 

def train_feature_extractor():
    base_model = InceptionV3(input_tensor=Input(shape=(224, 224, 3)), weights='imagenet', include_top=False) 
    x = base_model.output 
    x = GlobalAveragePooling2D()(x) 
    features = Dense(units=1024, activattion='selu', name='features')(x) 
    predictions = Dense(units=4251, activation='softmax')(features)  
    model = Model(inputs=base_model.input, outputs=predictions) 

    for layer in base_model.layers: 
        layer.trainable = False 

    model.compile(optimizer='nadam', loss='categorical_crossentropy') 

    train_datagen = ImageDataGenerator(**st.IMAGE_DATA_GENERATOR_TRAIN_KARGS) 
    test_datagen = ImageDataGenerator(**st.IMAGE_DATA_GENERATOR_TEST_KARGS) 

    train_generator = train_datagen.flow_from_directory(**st.FLOW_FROM_DIRECTORY_KARGS) 
    test_generator = test_datagen.flow_from_directory(**st.FLOW_FROM_DIRECTORY_KARGS) 

    model.fit_generator( 
        generator=train_generator, 
        steps_per_epoch=int(st.N_TRAINING_EXAMPLES/st.BATCH_SIZE), 
        epochs=st.EPOCHS, 
        verbose=1, 
        callbacks=[
            ModelCheckpoint(st.MODEL_CHECKPOINT_PATH, monitor='val_loss', save_best_only=True, verbose=1), 
            EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=1)
        ], 
        validation_data=test_generator, 
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
            ModelCheckpoint(st.MODEL_CHECKPOINT_PATH, monitor='val_loss', save_best_only=True, verbose=1), 
            EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=1)
        ], 
        validation_data=dg.test_generator, 
        validation_steps=int(st.N_TRAINING_EXAMPLES/st.BATCH_SIZE), 
        class_weight=None, 
        max_queue_size=10, 
        workers=1, 
        use_multiprocessing=True, 
        initial_epoch=0
    ) 

    return model 

train_feature_extractor()