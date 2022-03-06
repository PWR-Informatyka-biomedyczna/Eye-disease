from typing import Callable, Dict

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from dataset.dataset import EyeDiseaseData
from dataset.resamplers import identity_resampler


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
                 split_name: str = 'split',
                 batch_size: int = 16,
                 num_workers: int = 12,
                 shuffle_train: bool = True,
                 resampler: Callable = identity_resampler,
                 pretraining: bool = False,
                 binary=False,
                 k_folds: int = 10
                 ):
        super(EyeDiseaseDataModule, self).__init__()
        self.resampler: Callable = resampler
        self.pretraining = pretraining
        self.binary = binary
        # path
        self.csv_path: str = csv_path
        # split names
        self.train_split_name: str = train_split_name
        self.val_split_name: str = val_split_name
        self.test_split_name: str = test_split_name
        # transforms
        self.train_transforms: Compose = train_transforms
        self.val_transforms: Compose = val_transforms
        self.test_transforms: Compose = test_transforms
        # column names
        self.image_path_name: str = image_path_name
        self.target_name: str = target_name
        self.split_name = split_name
        # dataset parameters
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.shuffle_train: bool = shuffle_train
        # main dataframes
        self.data: Dict[str, pd.DataFrame] = {}
        self.k_folds = k_folds
        self.fold_idx = 0

    def prepare_data(self) -> None:
        df = self.resampler(pd.read_csv(self.csv_path))
        self.data['train'] = df[df[self.split_name] == self.train_split_name]
        self.data['val'] = df[df[self.split_name] == self.val_split_name]
        self.data['test'] = df[df[self.split_name] == self.test_split_name]
        
    def prepare_data_kfold(self) -> None:
        df = self.resampler(pd.read_csv(self.csv_path))
        self.data['train'] = df[df[self.split_name] == self.train_split_name]
        if not self.pretraining:
            if "val" in df[self.split_name].values:
                self.data['val'] = df[df[self.split_name] == self.val_split_name]
                self.data['val'][self.split_name] = 'train'
                self.data['train'] = pd.concat([self.data['train'], self.data['val']])
            self.data['test'] = df[df[self.split_name] == self.test_split_name]
        
        self.data['train'] = self.data['train'].reset_index(drop=True)
        skf = StratifiedKFold(n_splits=self.k_folds)
        self.fold_data = {"train": [], "val": []}
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(self.data['train'][self.target_name].values)
        for train_idx, val_idx in skf.split(self.data['train'], y):
            self.fold_data["train"].append(self.data['train'].loc[train_idx, :])
            self.fold_data["val"].append(self.data['train'].loc[val_idx, :])

    # def setup(self, stage: Optional[str] = None) -> None:
    #     pass

    def kfold_train_dataloader(self) -> DataLoader:
        train_data = self.fold_data['train'][self.fold_idx]
        return DataLoader(
            EyeDiseaseData(
                train_data,
                self.train_transforms,
                self.image_path_name,
                self.target_name,
                pretraining=self.pretraining,
                binary=self.binary
            ),
            shuffle=self.shuffle_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
    
    def kfold_val_dataloader(self) -> DataLoader:
        val_data  = self.fold_data['val'][self.fold_idx]
        return DataLoader(
            EyeDiseaseData(
                val_data,
                self.val_transforms,
                self.image_path_name,
                self.target_name,
                pretraining=self.pretraining,
                binary=self.binary
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def train_dataloader(self) -> DataLoader:
        """
        Prepares and returns train dataloader
        :return:
        """
        return DataLoader(
            EyeDiseaseData(self.data['train'],
                           self.train_transforms,
                           self.image_path_name,
                           self.target_name,
                           pretraining=self.pretraining,
                           binary=self.binary),
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
                           self.target_name,
                           pretraining=self.pretraining,
                           binary=self.binary),
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
                           self.target_name,
                           pretraining=self.pretraining,
                           binary=self.binary),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
