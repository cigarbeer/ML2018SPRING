import sys 
import os 

from util import * 
from setting import * 

if __name__ == '__main__': 
    training_data_path = sys.argv[1] 
    uid, mid, rating = read_training_data(training_data_path) 
    rating_normalized = normalize_rating(rating) 
    (n_users,) = uid.unique().shape 
    (n_movies,) = mid.unique().shape 
    model = get_matrix_factorization_model(n_users, n_movies, LATENT_DIMENSION) 
    train(model, X=[uid, mid], y=rating_normalized, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VALIDATION_SPLIT, model_checkpoint_path=MATRIX_FACTORIZATION_CHECKPOINT_PATH)
    
