from torch.optim import Adam,AdamW
from torch.utils.data import DataLoader
import os
from .dataset import DatasetMask4All,DatasetMask4AllPred
from .model import Model
from .modeling.unet_pp.unet_pp import UnetPlusPlus
from .transforms import TransformUnetPlusPlus



def load_model(config,device):

    '''
    Load the model imported from segmentation_models_pytorch
    '''
    name = config["model"]["name"]
    classes = config["model"]["classes"]
    
    if name == 'unet++':
        model = UnetPlusPlus(classes=classes)
    return model.to(device)

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


    return transformer

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
                              num_workers=os.cpu_count())
    
    val_loader = DataLoader(dataset=val_ds,
                            batch_size=batch_size,
                            shuffle=True)
    
    test_loader = DataLoader(dataset=test_ds,
                             batch_size=batch_size,
                             shuffle=False)
    
    return (train_loader,val_loader,test_loader)

def load_pred_dataset(images_dir,config):

    transform = load_transform(config)
    dataset = DatasetMask4AllPred(images_dir,transform)
    return dataset


