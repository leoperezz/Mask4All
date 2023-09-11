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

 
#Se necesita cargar 

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size','-tr',type=float,default=0.7)
    parser.add_argument('--val_size','-va',type=float,default=0.1)
    parser.add_argument('--test_size','-te',type=float,default=0.2)
    parser.add_argument('--config','-c',type=str,default="/home/lperez/code/Satellite/methods/DemoModel/configs/config_deepglob.json")

    args = parser.parse_args()

    return args

def apply_class_mapping(image:np.ndarray, 
                        dic1:Dict[str,List[int]], 
                        dic2:Dict[str,int]):
    
    result_image = np.zeros_like(image)
    for class_name, rgb_values in dic1.items():
        if class_name in dic2:
            class_value = dic2[class_name]
            mask = np.all(image == rgb_values, axis=-1)
            result_image[mask] = class_value
    return result_image[:,:,0]


def get_images_folders(images_names:List[str],
                       args:argparse.Namespace):

    '''
    Parameters:
    -----------

    images_names: Es una lista que contiene todos los nombres de las imágenes satélitales. 
                  Sus elementos son sus respectivos paths dentro de la pc. 
    '''

    number_images = len(images_names)

    train_size = int(number_images*args.train_size)
    val_size = int(number_images*args.val_size)
    test_size = int(number_images*args.test_size)

    train_images_path = images_names[:train_size]
    val_images_path = images_names[train_size:train_size+val_size]
    test_images_path = images_names[train_size+val_size:train_size+val_size+test_size]

    return (train_images_path,val_images_path,test_images_path)


def get_image_name(path_image:str):

    return path_image.split('/')[-1].split('_')[0]

def convert_image_to_mask_path(path_image:str,images_source_folder:str):

    '''
    Convierte el path de una imagen path/to/image_sat.jpg en path/to/image_mask.png 
    '''

    mask_name = path_image.split('/')[-1].split('_')[0]
    mask_name = mask_name+'_mask.png'
    return join(images_source_folder,mask_name)

def process_mask(mask_path:str,config_ds):

    mask_pil = Image.open(mask_path)
    mask_np = np.array(mask_pil)
    mask_processed = apply_class_mapping(mask_np,config_ds["label2rgb"],config_ds["label2id"])
    assert mask_processed.max() < 7, "[Error] Index out of bound"
    mask_pil = Image.fromarray(mask_processed)

    return mask_pil


def generate_data(images_paths:List[str],type_dataset:str,config):

    '''
    Realiza una copia de todos las imágenes que se encuentran en el folder de images_path y las guarda
    en el lugar de la carpeta del archivo constants.py unido con type_dataset.
    '''

    images_dest_folder = config["dataset"]["images_dest_folder"]
    images_source_folder = config["dataset"]["images_source_folder"]

    for image_path in tqdm(images_paths):
        
        name = get_image_name(image_path)
        
        mask_path = convert_image_to_mask_path(image_path,images_source_folder)
        
        if type_dataset == "test":
            mask_pil = Image.open(mask_path)
        else:
            mask_pil = process_mask(mask_path)
        image_pil = Image.open(image_path)

        name_image = join(images_dest_folder,type_dataset,"images",name+'.jpg')
        name_mask = join(images_dest_folder,type_dataset,"masks",name+'.png')

        image_pil.save(name_image)
        mask_pil.save(name_mask)



if __name__ == '__main__':

    #Set variables
    args = get_args()
    config = utils.load_config(args.config)
    images_names = glob(config["dataset"]["images_source_folder"]+'/*.jpg')

    #Get train,val,test images
    train_images_path,val_images_path,test_images_path = get_images_folders(images_names,args)

    #generate_data(train_images_path,"train",config)
    #generate_data(val_images_path,"val",config)
    generate_data(test_images_path,"test",config)





