from typing import Optional, Dict

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from dataset.dataset import EyeDiseaseData


class EyeDiseaseDataModule(pl.LightningDataModule):

    def __init__(self,
                 csv_path: str,
                 train_split_name: str = 'train',
                 val_split_name: str = 'val',
                 test_split_name: str = 'test',
                 train_transforms: Compose = ToTensor,
                 val_transforms: Compose = ToTensor,
                 test_transforms: Compose = ToTensor,
                 image_path_name: str = 'path',
                 target_name: str = 'label',
                 batch_size: int = 16,
                 num_workers: int = 12,
                 shuffle_train: bool = True
                 ):
        super(EyeDiseaseDataModule, self).__init__()
        self.csv_path: str = csv_path
        self.train_split_name: str = train_split_name
        self.val_split_name: str = val_split_name
        self.test_split_name: str = test_split_name
        self.train_transforms: Compose = train_transforms
        self.val_transforms: Compose = val_transforms
        self.test_transforms: Compose = test_transforms
        self.image_path_name: str = image_path_name
        self.target_name: str = target_name
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.shuffle_train: bool = shuffle_train
        self.data: Dict[str, pd.DataFrame] = {}

    def prepare_data(self) -> None:
        df = pd.read_csv(self.csv_path)
        self.data['train'] = df[self.train_split_name]
        self.data['val'] = df[self.val_split_name]
        self.data['test'] = df[self.test_split_name]

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        """
        Prepares and returns train dataloader
        :return:
        """
        return DataLoader(
            EyeDiseaseData(self.data['train'],
                           self.train_transforms,
                           self.image_path_name,
                           self.target_name),
            shuffle=self.shuffle_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        """
        Prepares and returns validate dataloader
        :return:
        """
        return DataLoader(
            EyeDiseaseData(self.data['val'],
                           self.val_transforms,
                           self.image_path_name,
                           self.target_name),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        """
        Prepares and returns test dataloader
        :return:
        """
        return DataLoader(
            EyeDiseaseData(self.data['test'],
                           self.test_transforms,
                           self.image_path_name,
                           self.target_name),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
