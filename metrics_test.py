import pytorch_lightning
import torch
import pandas as pd
import pytorch_lightning as pl
from dataset.dataset import EyeDiseaseData
from torchvision.transforms import Compose, ToTensor
from torchvision.datasets import CIFAR10
from dataset.transforms import train_transforms

data = CIFAR10('data', train=True, transform=train_transforms((224, 224)), download=True)

import random

import numpy as np
import cv2
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


from dataset.transforms import test_val_transforms, train_transforms
from methods.resnet import ResNetModel
from settings import LOGS_DIR, CHECKPOINTS_DIR
from training import train_test
from torch.utils.data import DataLoader
from pytorch_lightning.plugins.training_type.dp import DataParallelPlugin

dataset = DataLoader(data, batch_size=32, shuffle=True)

class Data(pl.LightningDataModule):

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
                 ):
        super(Data, self).__init__()

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
        #self.data: Dict[str, pd.DataFrame] = {}

    def prepare_data(self) -> None:
        pass
        #df = self.resampler(pd.read_csv(self.csv_path))
        #self.data['train'] = df[df[self.split_name] == self.train_split_name]
        #self.data['val'] = df[df[self.split_name] == self.val_split_name]
        #self.data['test'] = df[df[self.split_name] == self.test_split_name]

    # def setup(self, stage: Optional[str] = None) -> None:
    #     pass

    def train_dataloader(self) -> DataLoader:
        """
        Prepares and returns train dataloader
        :return:
        """
        return dataset

    def val_dataloader(self) -> DataLoader:
        """
        Prepares and returns validate dataloader
        :return:
        """
        return dataset

    def test_dataloader(self) -> DataLoader:
        """
        Prepares and returns test dataloader
        :return:
        """
        return dataset

# experiment setup
SEED = 0
PROJECT_NAME = 'ResnetTraining'
NUM_CLASSES = 10
LR = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 100
TARGET_SIZE = (224, 224)
NORMALIZE = True
MONITOR = 'val_loss'
PATIENCE = 10
GPUS = -1

models_list = [
        ResNetModel(NUM_CLASSES)
    ]


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


def main():
    seed_all(SEED)
    for model in models_list:
        data_module = Data('')

        hparams = {
            'dataset': type(data_module).__name__,
            'model_type': type(model).__name__,
            'lr': LR,
            'batch_size': BATCH_SIZE,
            'optimizer': 'Adam',
            'num_classes': NUM_CLASSES
        }

        logger = WandbLogger(
            save_dir=LOGS_DIR,
            config=hparams,
            project=PROJECT_NAME,
            log_model=False
        )

        callbacks = [
            EarlyStopping(
                monitor=MONITOR,
                patience=PATIENCE,
                mode='min'
            ),
            ModelCheckpoint(
                dirpath=CHECKPOINTS_DIR,
                save_top_k=1,
                monitor=MONITOR,
                mode='min'
            )
        ]
        train_test(
            model=model,
            datamodule=data_module,
            max_epochs=MAX_EPOCHS,
            num_classes=NUM_CLASSES,
            gpus=GPUS,
            lr=LR,
            callbacks=callbacks,
            logger=logger,
            strategy='dp'
        )
        logger.experiment.finish()


if __name__ == '__main__':
    main()
