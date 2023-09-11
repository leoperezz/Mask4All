import torch
from torchvision.transforms import ToPILImage

import logging
import wandb
import json
from PIL import Image
import numpy as np
from typing import Dict

from .functions import process_mask

def config_logging():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def config_wandb(config,name_project):
    experiment = wandb.init(project=name_project, resume='allow', anonymous='allow')
    experiment.config.update(
        dict(epochs=config["training"]["epochs"],
             batch_size=config["training"]["batch_size"],
             learning_rate=config["optimizer"]["learning_rate"])
    )
    return experiment

def load_config(path:str):

    with open(path,"r") as file:
        data = json.load(file)

    return data

def logging_info_epoch(output_train:Dict[str,float],
                       output_val:Dict[str,float],
                       experiment):
    
    experiment.log({
            "train loss":output_train["loss"],
            "train dice":output_train["dice"],
            "train iou":output_train["iou"],
            "val loss":output_val["loss"],
            "val dice":output_val["dice"],
            "val iou":output_val["iou"]
        })


    logging.info(f'''
                 train loss: {output_train["loss"]} 
                 train dice: {output_train["dice"]}
                 train iou: {output_train["iou"]}
                 val loss: {output_val["loss"]}
                 val dice: {output_val["dice"]}
                 val iou: {output_val["iou"]}
                 ''')

def get_wandb_image(image:Image.Image,caption):

    image = wandb.Image(image,
                        mode="RGB",
                        caption=caption)
    
    return image

def log_wandb_val_images(mask:torch.Tensor,pred:Image.Image,experiment,config):

    mask_pil = ToPILImage()(mask)
    size = config["training"]["resize"]
    original_size = (size,size)
    mask_pil = process_mask(np.array(mask_pil),original_size,config["dataset"])
    src_mask = get_wandb_image(mask_pil,"src mask")
    pred_img = get_wandb_image(pred,"pred mask")

    experiment.log({
        "src mask":src_mask,
        "pred img":pred_img
    })

