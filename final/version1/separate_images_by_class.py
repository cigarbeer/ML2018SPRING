import os 
import sys 
import shutil 

import pandas as pd 

import settings as st 

def read_image_class_mapping(file_name): 
    mapping = pd.read_csv(file_name) 
    return mapping 

def get_n_major_mapping(n, mapping): 
    mapping = pd.DataFrame()
    sorted_mapping = mapping.groupby(by=['Id']).count().sort_values(by=['Image'], ascending=False)
    n_major_ids = sorted_mapping.head(n)['Id'] 
    n_major_mapping = mapping[mapping['Id'].isin(n_major_ids)] 
    return n_major_mapping 

def safe_mkdir(path): 
    if os.path.exists(path): 
        return False 
    os.mkdir(path) 
    print('mkdir', path) 
    return  

def mkdir_by_ids(path, ids): 
    for Id in ids: 
        safe_mkdir(os.path.join(path, Id)) 
    return True 

def copy_images_to_corresponding_dir(src_path, dst_path, mapping): 
    for idx, (img_name, Id) in mapping.iterrows(): 
        shutil.copy(src=os.path.join(src_path, img_name), dst=os.path.join(dst_path, Id, img_name)) 
    return True 


if __name__ == '__main__': 
    src_dir = sys.argv[1] 
    dst_dir = sys.argv[2] 
    image_class_mapping_file = sys.argv[3] 
    top_n_class = int(sys.argv[4]) 
    if top_n_class == 0:
        top_n_class = st.N_CLASSES 

    mapping = read_image_class_mapping(image_class_mapping_file) 
    mapping = get_n_major_mapping(top_n_class, mapping) 
    safe_mkdir(dst_dir) 
    unique_ids = mapping.Id.unique() 
    mkdir_by_ids(dst_dir, unique_ids) 
    copy_images_to_corresponding_dir(src_dir, dst_dir, mapping)   


