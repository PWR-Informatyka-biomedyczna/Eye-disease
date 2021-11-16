from os import name
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from dataset.datamodule import EyeDiseaseDataModule
from dataset.transforms import test_val_transforms, train_transforms
from methods.resnet import ResNetModel

from settings import LOGS_DIR, PROJECT_DIR, CHECKPOINTS_DIR
from dataset import resamplers
from utils import train_test, ImagePredictionCallback

PROJECT_NAME = 'PROJECTTEST'
NUM_CLASSES = 2
LR = 1e-4
BATCH_SIZE = 2
TARGET_SIZE = (100, 100)
NORMALIZE = True
MONITOR = 'f1_score'


def main():
    model = ResNetModel(NUM_CLASSES)
    data_module = EyeDiseaseDataModule(
        csv_path=r'C:\Users\Konrad\Desktop\PythonProjects\Eye-disease\data\test.csv',
        train_split_name='train',
        val_split_name='val',
        test_split_name='test',
        train_transforms=train_transforms(TARGET_SIZE, NORMALIZE),
        val_transforms=test_val_transforms(TARGET_SIZE, NORMALIZE),
        test_transforms=test_val_transforms(TARGET_SIZE, NORMALIZE),
        image_path_name='Path',
        target_name='Label',
        split_name='Split',
        batch_size=BATCH_SIZE,
        num_workers=12,
        shuffle_train=True,
        resampler=resamplers.to_lowest_resampler(
            target_label='Label',
            train_split_name='train'
            )
    )
    data_module.prepare_data()

    hparams = {
        'dataset': type(data_module).__name__,
        'model_type': type(model).__name__,
        'lr': LR,
        'batch_size': BATCH_SIZE,
        'optimizer': 'adam',
        'num_classes': NUM_CLASSES
    }

    logger = WandbLogger(
        save_dir=LOGS_DIR,
        config=hparams,
        project=PROJECT_NAME
    )

    callbacks = [
    #    EarlyStopping(monitor=MONITOR)
    ]
    train_test(
        model=model,
        datamodule=data_module,
        max_epochs=100,
        num_classes=NUM_CLASSES,
        gpus=1,
        lr=LR,
        callbacks=callbacks,
        logger=logger
    )


if __name__ == '__main__':
    main()
