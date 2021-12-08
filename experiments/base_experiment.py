import random

import numpy as np
import cv2
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint



from dataset import EyeDiseaseDataModule, resamplers
from dataset.transforms import test_val_transforms, train_transforms
from methods import ResNet18Model, ResNet50Model
from settings import LOGS_DIR, CHECKPOINTS_DIR
from training import train_test


# experiment setup
SEED = 0
PROJECT_NAME = 'ResnetTraining'
NUM_CLASSES = 4
LR = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 100
TARGET_SIZE = (224, 224)
NORMALIZE = True
MONITOR = 'val_loss'
PATIENCE = 5
GPUS = -1

models_list = [
        #ResNetModel(NUM_CLASSES)
        ResNet50Model(NUM_CLASSES)
    ]


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


def main():
    seed_all(SEED)
    for model in models_list:
        data_module = EyeDiseaseDataModule(
            csv_path='/media/data/adam_chlopowiec/eye_image_classification/collected_data_splits.csv',
            train_split_name='train',
            val_split_name='val',
            test_split_name='test',
            train_transforms=train_transforms(TARGET_SIZE, NORMALIZE, cv2.INTER_NEAREST),
            val_transforms=test_val_transforms(TARGET_SIZE, NORMALIZE, cv2.INTER_NEAREST),
            test_transforms=test_val_transforms(TARGET_SIZE, NORMALIZE, cv2.INTER_NEAREST),
            image_path_name='Path',
            target_name='Label',
            split_name='Split',
            batch_size=BATCH_SIZE,
            num_workers=1,
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
            logger=logger
        )
        logger.experiment.finish()


if __name__ == '__main__':
    main()
