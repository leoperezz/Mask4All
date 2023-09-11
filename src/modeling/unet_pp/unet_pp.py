
import torch
from torch import nn
from torch import Tensor

import segmentation_models_pytorch as smp

from typing import Iterator,Tuple
from PIL import Image

import sys

sys.path.append("../../src")
from src.model import Model
from src.functions import process_mask

class UnetPlusPlus(Model):

    def __init__(self,classes:int):
        
        self.model = self._init_model(classes=classes)
    
    def _init_model(self,
                    classes:int,
                    in_channels:int=3,
                    encoder_name='timm-regnety_120',
                    encoder_weights='imagenet',
                    activation='softmax') -> nn.Module:
        
        return smp.UnetPlusPlus(encoder_name=encoder_name,
                                encoder_weights=encoder_weights,
                                activation=activation,
                                in_channels=in_channels,
                                classes=classes)

    def get_parameters_to_train(self) -> Iterator[nn.Parameter]:
        return self.model.parameters()
    
    def compute_loss(self, pred: Tensor, mask: Tensor) -> Tensor:
        '''
        Parameters:
        ----------
        pred(Tensor): (B,N,H,W)
        mask(Tensor): (B,H,W)

        '''
        loss_fn = smp.losses.DiceLoss(mode='multiclass',
                                      from_logits=False,
                                      ignore_index=0)
        return  loss_fn.forward(pred,mask)
    
    def forward(self,image: Tensor) -> Tensor:
        return self.model.forward(image)

    def postprocess(self,
                    pred: Tensor,
                    origin_size:Tuple[int,int],
                    config) -> Image.Image:

        pred = pred.argmax(dim=1).squeeze().detach().cpu().numpy()
        pred = process_mask(pred,origin_size,config["dataset"])
        return pred

    def load_ckpt(self,ckpt_path:str,device:torch.device):
        
        state_dict = torch.load(ckpt_path,map_location=device)["model"]
        self.model.load_state_dict(state_dict=state_dict)
        return self

    def to(self,device:torch.device):
        self.model.to(device)
        return self
    
    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()
