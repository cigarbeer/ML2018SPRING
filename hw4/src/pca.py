import numpy as np 
import skimage.io 
import os 
import sys 

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

def convert_image(img, X_average, U, k):  
    img = img - X_average 
    img_reduced = reduce_dimension(img, U, k) 
    img_recovered = recover_dimension(img_reduced, U, k) 
    img_result = img_recovered + X_average 
    return img_result 

def covert(input_file, output_file, X_average, U, k):
    img = read_image(input_file) 
    img = convert_image(img, X_average, U, k) 
    save_image(img, output_file) 
    return 


if __name__ == '__main__': 
    images_dir = sys.argv[1] 
    input_image_name = sys.argv[2] 
    input_image_file = os.path.join(images_dir, input_image_name) 
    output_file = sys.argv[3] 
    print('[read images from directory]')
    X = read_images_from_directory(images_dir) 
    print('[save average face]') 
    X_average = average_face(X) 
    save_image(X_average, 'average_face.jpg')  
    save_data(X_average, 'average_face.npy')
    print('[center data]') 
    X_centered = center_data(X)  
    print('[eigen value decomposition]') 
    eigenvector, eigenvalue = decompose(X_centered) 
    print('[save eigenvalue and eigenvector]') 
    save_data(eigenvector[:], 'eigenvector.npy') 
    save_data(eigenvalue, 'eigenvalue.npy') 
    print('[reduce dimension]') 
    X_reduced = reduce_dimension(X_centered, eigenvector, k=REDUCED_DIMENSION) 
    save_data(X_reduced, 'reduced.npy') 
    print('[recover dimension]') 
    X_recovered = recover_dimension(X_reduced, eigenvector, k=REDUCED_DIMENSION) 
    save_data(X_recovered, 'recovered.npy') 
    print('[read input image]') 
    img = read_image(input_image_file) 
    print('[center input image]') 
    img_centered = img - X_average 
    print('[reduce input image dimension]') 
    img_reduced = reduce_dimension(img_centered, eigenvector, k=REDUCED_DIMENSION) 
    print('[recover input image dimension]') 
    img_recovered = recover_dimension(img_reduced, eigenvector, k=REDUCED_DIMENSION) 
    print('[de-center recovered input image]') 
    img_decentered = img_recovered + X_average 
    print('[save results]') 
    save_image(img_decentered, output_file) 
    save_data(img_decentered, './reconstruction.npy') 
    print('[done]') 

# def read_data(file_path): 
#     return np.load(file_path) 

# def center_data(X):
#     mean = np.mean(X, axis=0) 
#     X = X - mean 
#     return X 

# def average_face(X):
#     return np.mean(X, axis=0)

# def svd(X):
#     U, S, Vh = np.linalg.svd(X.T) 
#     sorted_idx = np.argsort(np.diag(S)) 
#     U = U[:, sorted_idx]
#     S = S[:, sorted_idx]
#     return U, S

# def eigenvalue_to_pixel(e):
#     e -= np.min(e) 
#     e /= np.max(e) 
#     e = (e * 255).astype(np.uint8)
#     return e 

# def normalize(X): 
#     mu, sigma = statistics(X) 
#     X = (X -  mu) / sigma 
#     return X 

# def denormalize(N, mu, sigma): 
#     X = N * sigma + mu 
#     return X.astype(np.uint8)

# def statistics(X): 
#     mu = np.mean(X, axis=0) 
#     sigma = np.std(X, axis=0) + 1e-10
#     return mu, sigma 

# def reduce_dimension(X, U, k): 
#     return np.dot(X, U[:, :k]) 

# def recover_dimension(Z, U, k): 
#     return np.dot(Z, U[:, :k].T) 