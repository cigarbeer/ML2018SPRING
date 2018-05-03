import numpy as np 
import skimage.io 
import os 
import sys 
from sklearn.decomposition import PCA 
import pickle 

IMAGE_SHAPE = (600, 600, 3) 
REDUCED_DIMENSION = 4 

def read_images_from_directory(dir_name): 
    load_pattern = os.path.join(dir_name, '*.jpg') 
    img_names = skimage.io.collection.glob(load_pattern)
    img = skimage.io.imread_collection(img_names) 
    X = skimage.io.collection.concatenate_images(img) 
    m, *_ = X.shape
    X = X.reshape((m, -1)).T
    return X 

def read_image(file_name): 
    img = skimage.io.imread(file_name) 
    img = img.reshape((-1, 1)) 
    return img 

def center_data(X):
    return X - average_face(X) 

def average_face(X):
    return np.mean(X, axis=1).reshape((-1, 1))

def decompose(X): 
    U, S, V = np.linalg.svd(X, full_matrices=False) 
    eigenvector = U 
    eigenvalue = S 
    return eigenvector, eigenvalue 

def vector_to_pixel(e):
    e = e - np.min(e) 
    e = e / np.max(e) 
    e = (e * 255).astype(np.uint8)
    return e 

def save_data(x, file_name): 
    np.save(file_name, x) 
    return 

def read_data(file_name): 
    return np.load(file_name) 
 
def show_image(img): 
    skimage.io.imshow(vector_to_pixel(img.reshape(IMAGE_SHAPE))) 
    return 

def save_image(img, file_name): 
    skimage.io.imsave(file_name, vector_to_pixel(img.reshape(IMAGE_SHAPE))) 
    return 

def reduce_dimension(X, U, k): 
    return np.dot(X.T, U[:, :k]).T

def recover_dimension(Z, U, k): 
    return np.dot(Z.T, U[:, :k].T).T  


def save_average_face(img, path): 
    save_image(img, os.path.join(path, 'average_face.jpg')) 
    save_data(img, os.path.join(path, 'average_face.npy')) 
    return 


if __name__ == '__main__': 
    dir_name = sys.argv[1]
    X = read_images_from_directory(dir_name).T 
    pca = PCA(
        n_components=None, 
        copy=True, 
        whiten=False, 
        svd_solver='full', 
        tol=0.0, 
        iterated_power='auto', 
        random_state=None
    ) 
    pca.fit(X) 
    with open('pca.pickle', mode='w') as f: 
        pickle.dump(pca, f) 
        f.close() 
    
