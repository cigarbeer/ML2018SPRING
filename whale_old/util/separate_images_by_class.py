import os 
import sys 
import shutil 

import pandas as pd 

def read_image_class_mapping(file_name): 
    mapping = pd.read_csv(file_name) 
    return mapping 

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

    mapping = read_image_class_mapping(image_class_mapping_file) 
    safe_mkdir(dst_dir) 
    unique_ids = mapping.Id.unique() 
    mkdir_by_ids(dst_dir, unique_ids) 
    copy_images_to_corresponding_dir(src_dir, dst_dir, mapping)   


