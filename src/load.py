import os
import torch
from torch.optim import Adam,AdamW
from torch.utils.data import DataLoader

import sys
sys.path.append('/home/lperez/code/Satellite/methods/Mask4All')

from src.dataset import DatasetMask4All,DatasetMask4AllPred
from src.model import Model
from src.modeling.unet_pp.unet_pp import UnetPlusPlus
from src.modeling.mask2former.mask2former import Mask2Former
from src.transforms import (
    TransformUnetPlusPlus,
    TransformMask2Former)



def load_model(config,device):

    '''
    Load the model imported from segmentation_models_pytorch
    '''
    name = config["model"]["name"]
    
    if name == 'unet++':
        model = UnetPlusPlus(config,device)
    
    elif name == 'mask2former':
        model = Mask2Former(config,device)
    
    return model

def load_optim(config,model:Model):

    name = config["optimizer"]["name"]
    lr = config["optimizer"]["learning_rate"]

    if name == "adam":
        optim = Adam(model.get_parameters_to_train(),lr=lr)
    
    if name == "adamw":
        optim = AdamW(model.get_parameters_to_train(),lr=lr)
    
    return optim

def load_transform(config):

    name_model = config["model"]["name"]
    resize = config["training"]["resize"]
    
    if name_model == 'unet++':
        transformer = TransformUnetPlusPlus(resize)

    if name_model == 'mask2former':
        transformer = TransformMask2Former(resize)

    return transformer

def collate_fn(batch):
    return batch

def load_dataset(config):

    batch_size = config["training"]["batch_size"]
    images_dest_folder = config["dataset"]["images_dest_folder"]

    transform = load_transform(config)

    train_ds = DatasetMask4All(type_dataset='train',
                               data_dir=images_dest_folder,
                               transform=transform)

    val_ds = DatasetMask4All(type_dataset='val',
                               data_dir=images_dest_folder,
                               transform=transform)
    
    test_ds = DatasetMask4All(type_dataset='test',
                               data_dir=images_dest_folder,
                               transform=transform)

    train_loader = DataLoader(dataset=train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=os.cpu_count(),
                              collate_fn=collate_fn)
    
    val_loader = DataLoader(dataset=val_ds,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate_fn)
    
    test_loader = DataLoader(dataset=test_ds,
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=collate_fn)
    
    return (train_loader,val_loader,test_loader)

def load_pred_dataset(images_dir,config):

    transform = load_transform(config)
    dataset = DatasetMask4AllPred(images_dir,transform)
    return dataset




if __name__ == '__main__':

    from src.logg import load_config

    config = load_config('/home/lperez/code/Satellite/methods/Mask4All/configs/config_drone.json')
    train_loader,_,_ = load_dataset(config)
    for inputs in train_loader:
        print(inputs)