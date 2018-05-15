from . import data as D 

from keras.models import load_model 

import numpy as np 
import pandas as pd 

import sys 
import os 

def predict(model, generator): 
    predicted_probability = model.predict_generator( 
        generator=generator, 
        steps=1, 
        max_queue_size=10, 
        workers=os.cpu_count(), 
        use_multiprocessing=False, 
        verbose=1 
    ) 
    predicted_probability_sorted_reverse = np.argsort(predicted_probability, axis=1)[:, :-6:-1] 
    return predicted_probability_sorted_reverse  

def get_idx2class(class_indices): 
    idx2class = np.empty(shape=(len(class_indices),), dtype='object') 
    for class_name, idx in class_indices.items():
        idx2class[idx] = class_name 
    return idx2class 

# def write_prediction(prediction_idx, output_file, idx2class, ): 

if __name__ == '__main__': 
    model_path = sys.argv[1] 
    training_set_dir = sys.argv[2] 
    testing_set_dir = sys.argv[3] 
    prediction_output_file = sys.argv[4] 

    model = load_model(model_path) 
    idx2class = get_idx2class(D.get_train_generator(directory=training_set_dir).class_indices) 
    test_generator = D.get_test_generator(directory=testing_set_dir)
    prediction_idx = predict(model, test_generator) 
    prediction_class_name = idx2class[prediction_idx] 
    class_name_str = pd.DataFrame(prediction_class_name).apply(lambda x: ' '.join(x), axis=1) 
    image_name_str = pd.DataFrame(test_generator.filenames).apply(lambda x: os.path.basename(x)) 
    df = pd.concat([image_name_str, class_name_str], names=['Image', 'Id']) 
    df.to_csv(prediction_output_file)
