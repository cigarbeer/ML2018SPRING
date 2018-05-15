from keras.preprocessing import image 

def get_train_generator(directory)
    train_generator = image.ImageDataGenerator(
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
    ).flow_from_directory(
        directory=directory, 
        target_size=(224, 224), 
        color_mode='rgb', 
        classes=None, 
        class_mode='categorical', 
        batch_size=256, 
        shuffle=True, 
        seed=0, 
        save_to_dir=None, 
        save_prefix='', 
        save_format='png', 
        follow_links=False
    ) 
    return train_generator 


def get_test_generator(directory): 
    test_generator = image.ImageDataGenerator(
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
    ).flow_from_directory(
        directory=directory, 
        target_size=(224, 224), 
        color_mode='rgb', 
        classes=None, 
        class_mode=None, 
        batch_size=1, 
        shuffle=False, 
        seed=0, 
        save_to_dir=None, 
        save_prefix='', 
        save_format='png', 
        follow_links=False
    ) 
    return test_generator 