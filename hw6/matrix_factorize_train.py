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
    (n_users,) = uid.unique().shape 
    (n_movies,) = mid.unique().shape 
    print('[build matrix factorization model]')
    model = get_matrix_factorization_model(matrix_shape=(n_users, n_movies), latent_dimension=LATENT_DIMENSION) 
    print('[start training]')
    train(model, X=[uid, mid], y=rating_normalized, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VALIDATION_SPLIT, model_checkpoint_path=MATRIX_FACTORIZATION_CHECKPOINT_PATH) 
    print('[done]') 

