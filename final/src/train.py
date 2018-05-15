from . import data as D 
from . import model as M 

from keras.optimizers import Nadam 

if __name__ == '__main__': 
    for layer in M.vgg16.layers:
        layer.trainable = False 
    M.model.compile(
        optimizer=Nadam(lr=0.001), 
        loss='categorical_crossentropy'
    ) 
    M.model.fit_generator(
        
    )