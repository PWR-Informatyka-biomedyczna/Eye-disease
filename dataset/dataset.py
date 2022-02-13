from typing import Dict, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset

import pandas as pd
from torchvision.transforms import Compose


class EyeDiseaseData(Dataset):

    def __init__(self,
                 df: pd.DataFrame,
                 transforms: Compose,
                 image_path_name: str = 'path',
                 label_name: str = 'target',
                 pretraining: bool = False,
                 binary: bool = False):
        self.data = df
        self.transforms = transforms
        self.path_name = image_path_name
        self.label_name = label_name
        self.pretraining = pretraining
        self.binary = binary

    def __len__(self) -> int:
        return self.data.__len__()

    def _process_image(self, image_path: str) -> torch.Tensor:
        img = Image.open(image_path)
        return self.transforms(img)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        path, label = row[self.path_name], row[self.label_name]
        image = self._process_image(path)
        if self.pretraining:
            if label == 3:
                label = 1
                
        if self.binary:
            if label > 1:
                label = 1
                
        return {'input': image, 'target': label}
