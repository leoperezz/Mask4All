import torch
from torch import Tensor
from torch import nn

import numpy as np
from PIL import Image
from typing import Dict,List

from .model import Model

def get_values_metrics(pred:Tensor,
                       mask:Tensor,
                       metrics:dict):

    dice_value = metrics["dice"](pred,mask)
    iou_value = metrics["iou"](pred,mask)

    return dice_value,iou_value

def generate_mask(inputs):
    masks = []
    for i in inputs:
        masks.append(i["mask"].unsqueeze(dim=0))
    masks = torch.cat(masks,dim=0)
    return masks

def generate_images(inputs):
    images = []
    for i in inputs:
        images.append(i["image"].unsqueeze(dim=0))
    images = torch.cat(images,dim=0)
    return images

def train_step(model:Model,
               optim:torch.optim.Optimizer,
               inputs:Dict[str,Tensor],
               metrics:dict):

    optim.zero_grad()

    mask = generate_mask(inputs)

    output = model.forward(inputs)

    pred = output['mask_predicted']
    loss = output['loss']

    loss.backward()

    nn.utils.clip_grad_norm_(model.get_parameters_to_train(), 1.0)
    
    optim.step()

    dice_value,iou_value = get_values_metrics(pred,mask,metrics)
    
    return loss.item(),dice_value.item(),iou_value.item()

def val_step(model:Model,
             inputs:Dict[str,Tensor],
             metrics:dict):
    
    mask = generate_mask(inputs)

    output = model.forward(inputs)

    pred = output['mask_predicted']
    loss = output['loss']

    dice_value,iou_value = get_values_metrics(pred,mask,metrics)
    
    return loss.item(),dice_value.item(),iou_value.item()

def generate_checkpoint(model:Model,
                        optim:torch.optim.Optimizer):
    
    ckpt = dict(model=model.model.state_dict(),optim=optim.state_dict())

    return ckpt

def map_values_to_rgb(image:np.ndarray, 
                      dic1:Dict[str,List[int]],
                      dic2:Dict[str,int]):
    
    h, w = image.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    for value, category in dic2.items():
        mask = (image == category)
        rgb_values = dic1[value]
        rgb_image = np.where(np.expand_dims(mask, axis=-1), rgb_values, rgb_image)
    return rgb_image

def process_mask(pred:np.ndarray,
                 original_size:tuple,
                 config):
    
    '''
    Convierte una imagen en array en un tamaño fijo y convierte sus pixeles a RGB.
    '''

    pred_pil = Image.fromarray(pred.astype(np.uint8)).resize(original_size,resample=Image.Resampling.NEAREST)
    output = np.array(pred_pil)
    output = map_values_to_rgb(output,config["label2rgb"],config["label2id"])
    output = Image.fromarray(output.astype(np.uint8))

    return output
    
def predict_single_image(image:torch.Tensor,
                         model:Model,
                         device:torch.device,
                         origin_size:int,
                         config)->Image.Image:
    
    image = image.unsqueeze(0).to(device)
    pred = model.forward(image)
    pred = pred.argmax(dim=1).squeeze().detach().cpu().numpy()
    original_size = (origin_size,origin_size)
    pred = process_mask(pred,original_size,config)

    return pred