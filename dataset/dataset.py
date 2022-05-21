from typing import Dict, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset

import pandas as pd
import torchvision
from torchvision.transforms import Compose
from dataset.transforms import test_val_transforms


class EyeDiseaseData(Dataset):

    def __init__(self,
                 df: pd.DataFrame,
                 transforms: Compose,
                 image_path_name: str = 'path',
                 label_name: str = 'target',
                 pretraining: bool = False,
                 binary: bool = False,
                 unlabeled: bool = False):
        self.data = df
        self.transforms = transforms
        self.path_name = image_path_name
        self.label_name = label_name
        self.pretraining = pretraining
        self.binary = binary
        self.unlabeled = unlabeled
        self.test_val_transforms = test_val_transforms(target_size=(224, 224), normalize=True)

    def __len__(self) -> int:
        return self.data.__len__()

    def _process_image(self, image_path: str) -> torch.Tensor:
        img = Image.open(image_path)
        return self.transforms(img)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        path, label = row[self.path_name], row[self.label_name]
        if self.unlabeled:
            image = Image.open(path)
            img_aug = self._process_image(path)
            image = self.test_val_transforms(image)
        else:
            image = self._process_image(path)
        if self.pretraining:
            if label == 3:
                label = 1
                
        if self.binary:
            if label > 1:
                label = 1
        
        return {'input': image, 'target': label}
        # if self.unlabeled:
        #     return (image, img_aug), label
        # else:
        #     return image, label
