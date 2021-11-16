from os import name
import torch
import pytorch_lightning as pl
from dataset.datamodule import EyeDiseaseDataModule
from dataset.transforms import test_val_transforms, train_transforms
from methods.resnet import ResNetModel

from settings import LOGS_DIR, PROJECT_DIR, CHECKPOINTS_DIR


PROJECT_NAME = 'PROJECTTEST'
LR = 1e-4
BATCH_SIZE = 1
TARGET_SIZE = (100, 100)
NORMALIZE = True


def main():
    model = ResNetModel(2)
    data_module = EyeDiseaseDataModule(
        csv_path = '',
        train_split_name='train',
        val_split_name = 'val',
        test_split_name='test',
        train_transforms=train_transforms(TARGET_SIZE, NORMALIZE),
        val_transforms=test_val_transforms(TARGET_SIZE, NORMALIZE),
        test_transforms=test_val_transforms(TARGET_SIZE, NORMALIZE),
        image_path_name='Path',
        target_name='Label',
        batch_size=BATCH_SIZE,
        num_workers=12,
        shuffle_train=True
    )

    hparams = {
        'dataset': type(data_module).__name__,
        'model_type': type(model).__name__,
        'lr': LR,
        'batch_size': BATCH_SIZE,
        'optimizer': 'adam'
    }

    logger = pl.loggers.WandbLogger(
        save_dir=LOGS_DIR,
        config=hparams,
        project_name=PROJECT_NAME
    )

if __name__ == '__main__':
    main()
