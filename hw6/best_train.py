import sys 
import os 

from util import * 
from setting import * 

if __name__ == '__main__': 
    training_data_path = sys.argv[1] 
    print('[read training data]')
    uid, mid, rating = read_training_data(training_data_path) 
    print('[normalize rating]')
    rating_normalized = normalize_rating(rating) 
    print('[get number of users and movies]')
    n_users = uid.max() 
    n_movies = mid.max() 
    print('[build submodel 1]') 
    submodel = get_best_model(matrix_shape=(n_users, n_movies), latent_dimension=LATENT_DIMENSION) 
    print('[suffle index 1]')
    uid, mid, rating = read_training_data(training_data_path) 
    print('[start training submodel 1]')
    train(submodel, X=[uid, mid], y=rating_normalized, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VALIDATION_SPLIT, model_checkpoint_path=SUBMODEL_1_PATH) 
    print('[build submodel 2]')
    submodel = get_best_model(matrix_shape=(n_users, n_movies), latent_dimension=LATENT_DIMENSION) 
    print('[suffle index 2]')
    uid, mid, rating = read_training_data(training_data_path) 
    print('[start training submodel 2]')
    train(submodel, X=[uid, mid], y=rating_normalized, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VALIDATION_SPLIT, model_checkpoint_path=SUBMODEL_2_PATH) 
    print('[build submodel 3]')
    submodel = get_best_model(matrix_shape=(n_users, n_movies), latent_dimension=LATENT_DIMENSION) 
    print('[suffle index 3]')
    uid, mid, rating = read_training_data(training_data_path) 
    print('[start training submodel 3]')
    train(submodel, X=[uid, mid], y=rating_normalized, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VALIDATION_SPLIT, model_checkpoint_path=SUBMODEL_3_PATH) 
    print('[build ensemble model]') 
    model = get_ensemble_model()
    print('[done]') 

