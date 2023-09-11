import os
from os.path import join
from glob import glob
from shutil import copy
import argparse
from tqdm import tqdm
from typing import List,Dict

import numpy as np
from PIL import Image

import sys
sys.path.append('/home/lperez/code/Satellite/methods/DemoModel')
from src import utils

 
#Se necesita cargar config_satimgseg.json

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size','-tr',type=float,default=0.7)
    parser.add_argument('--val_size','-va',type=float,default=0.1)
    parser.add_argument('--test_size','-te',type=float,default=0.2)
    parser.add_argument('--config','-c',type=str,default="../configs/config_satimgseg.json")

    args = parser.parse_args()

    return args

def split_validation(validation_images_path:List[str],
                     validation_masks_path:List[str],
                     val_size:float=0.6):
    
    validation_images_path.sort()
    validation_masks_path.sort()

    total_images = len(validation_images_path)
    validation_num = int(total_images*val_size)

    val_images_path = validation_images_path[:validation_num]
    test_images_path = validation_images_path[validation_num:]

    val_masks_path = validation_masks_path[:validation_num]
    test_masks_path = validation_masks_path[validation_num:]

    return dict(val_images=val_images_path,
                test_images=test_images_path,
                val_masks=val_masks_path,
                test_masks=test_masks_path)

def get_mask_name(path_mask:str):

    name = path_mask.split('/')[-1].split('.')[0]
    idx = name.find("15label")
    name = name[:idx]+name[idx+8:]
    return name

def get_image_name(path_image:str):

    return path_image.split('/')[-1].split('.')[0]

def open_tif_image(path_image:str):

    img = Image.open(path_image)

    return img

def generate_data(data_path:Dict[str,str],type_dataset:str,config):

    '''
    Realiza una copia de todos las imágenes que se encuentran en el folder de images_path y las guarda
    en el lugar de la carpeta del archivo constants.py unido con type_dataset.
    '''

    dest_folder = join(config["dataset"]["images_dest_folder"],type_dataset)

    images_path = data_path[f"{type_dataset}_images"]
    masks_path = data_path[f"{type_dataset}_masks"]

    paths = zip(images_path,masks_path)

    for image_p,mask_p in tqdm(paths,total=len(images_path)):
        
        image_name = get_image_name(image_p)
        mask_name = get_mask_name(mask_p)
    
        assert image_name == mask_name,(image_p,mask_p)

        image_pil = open_tif_image(image_p)
        mask_pil = open_tif_image(mask_p)

        mask_np = np.array(mask_pil)[:,:,0]
        assert mask_np.max() < 16, mask_np.max()
        if type_dataset == 'test':
            mask_np = utils.map_values_to_rgb(mask_np,
                                              config["dataset"]["label2rgb"],
                                              config["dataset"]["label2id"])
        mask_pil = Image.fromarray(mask_np.astype(np.uint8))

        

        image_pil.save(join(dest_folder,"images",image_name+'.jpg'))
        mask_pil.save(join(dest_folder,"masks",mask_name+'.png'))


#GF2_PMS1__L1A0001064454-MSS1_00.tif




if __name__ == '__main__':

    #Set variables
    args = get_args()
    config = utils.load_config(args.config)
    
    train_images_path = glob(config["dataset"]["images_source_folder"]+'/train_images/train/*.tif')
    train_masks_path = glob(config["dataset"]["images_source_folder"]+'/train_masks/train/*.tif')

    print(len(train_images_path))
    print(len(train_masks_path))
    
    train_images_path.sort()
    train_masks_path.sort()

    assert len(train_images_path) == len(train_masks_path)

    validation_images_path = glob(config["dataset"]["images_source_folder"]+'/val_images/val/*.tif')
    validations_masks_path = glob(config["dataset"]["images_source_folder"]+"/val_masks/val/*.tif")

    assert len(validation_images_path) == len(validations_masks_path)

    data_path = split_validation(validation_images_path,validations_masks_path,val_size = 0.6)
    data_path["train_images"]=train_images_path
    data_path["train_masks"]=train_masks_path

    #print(data_path)
    #generate_data(data_path,"train",config)
    generate_data(data_path,"test",config)
    #generate_data(data_path,"val",config)




