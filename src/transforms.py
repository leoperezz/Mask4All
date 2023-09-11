import torch
from torch import Tensor
from torchvision.transforms import PILToTensor,Normalize


from abc import ABC,abstractmethod
from PIL import Image
from typing import Tuple


class Transform:

    def __init__(self,resize:int):

        self.resize_shape = (resize,resize)
    
    @abstractmethod
    def transform(self,image_path,mask_path) -> Tuple[Tensor,Tensor]:
        pass

    @abstractmethod
    def path_to_tensor(self,path:str,is_mask:bool) -> Tensor:
        pass
    

class TransformUnetPlusPlus(Transform):

    def __init__(self,resize:int):

        super(TransformUnetPlusPlus,self).__init__(resize)

    def transform(self,image_path,mask_path=None) -> Tuple[Tensor,Tensor]:

        image_pt = self.path_to_tensor(image_path,False)
        mask_pt = self.path_to_tensor(mask_path,True)

        return (image_pt,mask_pt)
    
    def path_to_tensor(self,path:str,is_mask:bool)->Tensor:

        if is_mask:
            resampling = Image.Resampling.NEAREST
            image = Image.open(path).resize(self.resize_shape,resample=resampling)
            image = PILToTensor()(image).squeeze()
        else:
            resampling = Image.Resampling.BILINEAR
            image = Image.open(path).resize(self.resize_shape,resample=resampling)
            image = PILToTensor()(image).type(torch.float32)
            image = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])(image)
        return image




