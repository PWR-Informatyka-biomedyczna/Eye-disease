import torch.utils.data
import pandas as pd
from typing import Dict

from torchvision.transforms import Compose
from PIL import Image


class SearchDataset(torch.utils.data.Dataset):

    def __init__(self,
                 transforms: Compose,
                 image_path_name: str = 'Path',
                 label_name: str = 'Label'):
        self.data = pd.read_csv('/media/data/adam_chlopowiec/eye_image_classification/resized_224x224_collected_data_splits.csv')
        self.data = self.data[self.data['Split'] == 'train']
        self.transforms = transforms
        self.path_name = image_path_name
        self.label_name = label_name

    def __len__(self) -> int:
        return self.data.__len__()

    def _process_image(self, image_path: str) -> torch.Tensor:
        img = Image.open(image_path)
        if self.transforms is not None:
            return self.transforms(image=img)['image']
        return img

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        path, label = row[self.path_name], row[self.label_name]
        image = self._process_image(path)
        return image, label
