
import torch
from torch import nn,Tensor

from typing import (Iterator,
                    Dict,
                    List,
                    Tuple)

from transformers import (
    Mask2FormerForUniversalSegmentation,
    AutoImageProcessor)

from PIL import Image
import numpy as np

import sys
sys.path.append('/home/lperez/code/Satellite/methods/Mask4All/')
from src.model import Model

import json

class Mask2Former(Model):

    def __init__(self,config,device:torch.device):
        
        self.config = config
        self.device = device
        self.model = self._init_model(config)
        self.model.to(device)
    
    def _init_model(self,config) -> Mask2FormerForUniversalSegmentation:
        
        label2id = self.modify_label2id(config["dataset"]["label2id"])
        id2label = dict(zip(label2id.values(),label2id.keys()))
        
        ckpt = "facebook/mask2former-swin-small-coco-instance"
        processor = AutoImageProcessor.from_pretrained(ckpt,label2id=label2id)
        self.processor = processor

        return Mask2FormerForUniversalSegmentation.from_pretrained(
            ckpt,
            label2id = label2id,
            id2label = id2label,
            ignore_mismatched_sizes=True)

    def get_parameters_to_train(self) -> Iterator[nn.Parameter]:
        return self.model.parameters()

    def compute_loss(self, pred: Tensor, mask: Tensor) -> Tensor:
        return super().compute_loss(pred, mask)
    
    def forward(self, inputs: Dict[str,Tensor]) -> Dict[str,Tensor]:
        
        inputs = self.process_input_train(inputs)

        b,h,w = inputs["mask"].shape
        target_sizes = [(h,w) for i in range(b)]
        output = self.model(pixel_values=inputs["image"],
                            class_labels=inputs["class_labels"],
                            mask_labels=inputs["mask_labels"])
        
        loss = output.loss
        pred = self.processor.post_process_semantic_segmentation(output,
                                                                 target_sizes=target_sizes)
        
        pred = torch.cat([i.unsqueeze(0) for i in pred],dim=0).cpu()
        return dict(loss=loss,mask_predicted=pred)
    
    def predict(self,image: Tensor,original_size:Tuple[int,int]) -> Image.Image:
        '''
        Solo hace la predicción para una sola imagen

        image : (1,C,H,W)

        original_size: El size al que será re escalado.

        '''

        output = self.model(pixel_values=image.to(self.device))

        pred = self.processor.post_process_semantic_segmentation(output,
                                                                 target_sizes=[original_size])

        pred = pred[0].cpu().numpy()
        pred_pil = Image.fromarray(pred.astype(np.uint8)).resize(original_size,
                                                                 resample=Image.Resampling.BILINEAR)
        output = np.array(pred_pil)
        output = self.map_values_to_rgb(output,
                                        self.config["dataset"]["label2rgb"],
                                        self.config["dataset"]["label2id"])
        output = Image.fromarray(output.astype(np.uint8))
        
        return output

    def process_input_train(self,inputs):

        images,masks,class_labels,mask_labels = [],[],[],[]

        for i in inputs:
            images.append(i["image"].unsqueeze(dim=0))
            masks.append(i["mask"].unsqueeze(dim=0))
            class_labels.append(i["class_labels"].to(self.device))
            mask_labels.append(i["mask_labels"].to(self.device))

        images = torch.cat(images,dim=0).to(self.device)
        masks = torch.cat(masks,dim=0)

        return dict(
            image = images,
            mask = masks,
            class_labels = class_labels,
            mask_labels = mask_labels
        )


    def map_values_to_rgb(self,
                          image:np.ndarray, 
                          dic1:Dict[str,List[int]],
                          dic2:Dict[str,int]):
    
        h, w = image.shape
        rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    
        for value, category in dic2.items():
            mask = (image == category)
            rgb_values = dic1[value]
            rgb_image = np.where(np.expand_dims(mask, axis=-1), rgb_values, rgb_image)
        return rgb_image


    def to(self, device: torch.device):
        self.model.to(device)
        return self

    def modify_label2id(self,label2id:Dict[str,int]):

        keys = list(label2id.keys())[1:]
        values = [i for i in range(len(keys))]
        label2id = dict(zip(keys,values))
        return label2id

if __name__ == '__main__':

    def load_config(path:str):

        with open(path,"r") as file:
            data = json.load(file)

        return data

    config = load_config('/home/lperez/code/Satellite/methods/Mask4All/configs/config_drone.json')
    model = Mask2Former(config)
    print(model.processor)