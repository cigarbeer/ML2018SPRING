from keras.applications.inception_v3 import InceptionV3 

from keras.preprocessing import image 
from keras.preprocessing.image import ImageDataGenerator 

from keras.models import Model 
from keras.models import load_model 

from keras.layers import Dense 
from keras.layers import GlobalAveragePooling2D 
from keras.layers import Input 

from keras.callbacks import EarlyStopping 
from keras.callbacks import ModelCheckpoint 
from keras.callbacks import ReduceLROnPlateau 

import settings as st 

def train_feature_extractor(train_generator, test_generator):
    base_model = InceptionV3(input_tensor=Input(shape=(224, 224, 3)), weights='imagenet', include_top=False) 
    x = base_model.output 
    x = GlobalAveragePooling2D()(x) 
    x = Dense(units=st.FEATURE_DIM, activation='selu', name='features')(x) 
    features = Dense(units=st.FEATURE_DIM, activation='selu', name='features')(x) 
    predictions = Dense(units=st.N_CLASSES, activation='softmax')(features)  
    model = Model(inputs=base_model.input, outputs=predictions) 

    for layer in base_model.layers: 
        layer.trainable = False 

    model.compile(optimizer='nadam', loss='categorical_crossentropy') 

    # train_datagen = ImageDataGenerator(**st.IMAGE_DATA_GENERATOR_TRAIN_KARGS) 
    # test_datagen = ImageDataGenerator(**st.IMAGE_DATA_GENERATOR_TEST_KARGS) 

    # train_generator = train_datagen.flow_from_directory(directory=training_set_directory, **st.FLOW_FROM_DIRECTORY_KARGS) 
    # test_generator = test_datagen.flow_from_directory(directory=training_set_directory, **st.FLOW_FROM_DIRECTORY_KARGS) 

    model.fit_generator( 
        generator=train_generator, 
        steps_per_epoch=int(st.N_TRAINING_EXAMPLES/st.BATCH_SIZE), 
        epochs=st.EPOCHS, 
        verbose=1, 
        callbacks=[
            ModelCheckpoint(st.FEATURE_EXTRACTOR_PATH, monitor='val_loss', save_best_only=True, verbose=1), 
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
        generator=train_generator, 
        steps_per_epoch=int(st.N_TRAINING_EXAMPLES/st.BATCH_SIZE), 
        epochs=st.EPOCHS, 
        verbose=1, 
        callbacks=[
            ModelCheckpoint(st.FEATURE_EXTRACTOR_PATH, monitor='val_loss', save_best_only=True, verbose=1), 
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

    return model 

def load_feature_extractor():   
    model = load_model(st.FEATURE_EXTRACTOR_PATH) 
    feature_extractor = Model(inputs=model.input, outputs=model.get_layer('features').output) 
    return feature_extractor 