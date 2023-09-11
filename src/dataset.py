from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor


from os import listdir
from os.path import join
from typing import Tuple
from PIL import Image
import numpy as np

from .transforms import Transform


class DatasetMask4All(Dataset):

    def __init__(self,
                 type_dataset:str,
                 data_dir:str,
                 transform:Transform):
        
        self.transform = transform

        self.image_dir_path = join(data_dir,type_dataset,'images')
        self.mask_dir_path = join(data_dir,type_dataset,'masks')
        self.images_dir = listdir(self.image_dir_path)
        self.masks_dir = listdir(self.mask_dir_path)

        assert len(self.images_dir) == len(self.masks_dir)

        self.images_dir.sort()
        self.masks_dir.sort()
    
    def __len__(self):
        return len(self.images_dir)

    def __getitem__(self, idx:int)->Tuple[Tensor,Tensor]:

        image_path = join(self.image_dir_path,self.images_dir[idx])
        mask_path = join(self.mask_dir_path,self.masks_dir[idx])

        image_pt,mask_pt = self.transform.transform(image_path,mask_path)

        return (image_pt,mask_pt)
    
class DatasetMask4AllPred(Dataset):

    def __init__(self,
                 data_dir:str,
                 transform:Transform):
        
        self.images_dir = listdir(data_dir)
        self.transform = transform

        self.images_dir.sort()

    def __len__(self) -> int:
        return len(self.images_dir)

    def __getitem__(self, idx) -> Tensor:
        
        image_name = self.images_dir[idx]
        image_path = join(self.images_dir,image_name)
        origin_size = self._get_original_size(image_path)
        image_pt = self.transform.path_to_tensor(image_path,False).unsqueeze(0)
        return image_pt,origin_size

    def _get_original_size(self,path) -> Tuple[int,int]:
        h,w,c = np.array(Image.open(path))
        return (h,w)








