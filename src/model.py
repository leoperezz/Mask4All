from abc import ABC,abstractmethod


import torch
from torch import nn
from torch import Tensor

from PIL import Image
from typing import Iterator


class Model:

    def __init__(self):

        self.model = self._init_model()
    
    @abstractmethod
    def _init_model(self)->nn.Module:
        pass

    def freeze_parameters(self):
        pass

    @abstractmethod
    def get_parameters_to_train(self) -> Iterator[nn.Parameter]:
        pass

    @abstractmethod
    def compute_loss(self,pred:Tensor,mask:Tensor)->Tensor:
        pass

    @abstractmethod
    def forward(self,image:Tensor) -> Tensor:
        pass

    @abstractmethod
    def postprocess(self,pred:Tensor) -> Image.Image:
        pass

    @abstractmethod
    def to(self,device:torch.device):
        pass


