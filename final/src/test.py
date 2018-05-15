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
    training_set_dir = sys.argv[2] 
    testing_set_dir = sys.argv[3] 
    prediction_output_file = sys.argv[4] 

    model = load_model(model_path) 
    prediction = predict(model, D.get_test_generator(directory=testing_set_dir)) 