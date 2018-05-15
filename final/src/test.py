from . import data as D 

from keras.models import load_model 

import sys 
import os 

def predict(model, generator): 
    prediction = model.predict_generator( 
        generator=generator, 
        steps=1, 
        max_queue_size=10, 
        workers=os.cpu_count(), 
        use_multiprocessing=True, 
        verbose=1 
    ) 
    return prediction 






if __name__ == '__main__': 
    model_path = sys.argv[1] 
    model = load_model(model_path) 
    prediction = predict(model, D.test_generator) 