import time
import hashlib
import random
import os
import datetime
from pathlib import Path
from methods.classifier import Classifier

import numpy as np
import cv2
import torch
import torchvision
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint



from dataset import EyeDiseaseDataModule, resamplers
from dataset.transforms import test_val_transforms, train_transforms
from methods import ResNet18Model, ResNet50Model, EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, Xception
from methods import DenseNet, ResNext50, ResNext101
from settings import LOGS_DIR, CHECKPOINTS_DIR, PROJECT_DIR
from training import train_test

# PL_TORCH_DISTRIBUTED_BACKEND=gloo poetry run python3 -m experiments.base_experiment
# experiment setup
SEED = 0
PROJECT_NAME = 'ResNetTraining'
NUM_CLASSES = 4
LR = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 100
NORMALIZE = True
MONITOR = 'val_loss'
PATIENCE = 5
GPUS = -1
ENTITY_NAME = 'kn-bmi'
RESAMPLER = resamplers.identity_resampler
WEIGHTS = torch.Tensor([1, 2, 2.5, 1.5])
TYPE = 'test' # pretraining, training, training-from-pretrained
MODEL_PATH = '/home/adam_chlopowiec/data/eye_image_classification/Eye-disease/checkpoints/training/ResNet18Model/2021-12-16_16:22:11.605147/ResNet18Model.ckpt'
TEST_ONLY = True
DATE_NOW = str(datetime.datetime.now())

models_list = [
        #EfficientNetB0(NUM_CLASSES),
        #EfficientNetB2(NUM_CLASSES),
        #Xception(NUM_CLASSES),
        #DenseNet(NUM_CLASSES),
        #ResNext50(NUM_CLASSES),
        ResNet18Model(NUM_CLASSES),
        #ResNet50Model(NUM_CLASSES)
    ]


def load_model(model, mode: str = 'train'):
    classifier = Classifier(
                model=model,
                num_classes=NUM_CLASSES,
                lr=LR,
                optimizer=torch.optim.Adam,
                weights=WEIGHTS
                )
    in_features = model.get_last_layer().in_features
    out_features = model.get_last_layer().out_features
    if mode == 'train':
        if out_features > 2:
            model.set_last_layer(torch.nn.Linear(in_features, 2))
        classifier.load_from_checkpoint(checkpoint_path=MODEL_PATH, model=model, num_classes=NUM_CLASSES, lr=LR, weights=WEIGHTS[:2])
        classifier.model.set_last_layer(torch.nn.Linear(in_features, NUM_CLASSES))
        classifier.criterion.weight = WEIGHTS
    elif mode == 'test':
        if out_features < NUM_CLASSES:
            model.set_last_layer(torch.nn.Linear(in_features, NUM_CLASSES))
        classifier.load_from_checkpoint(checkpoint_path=MODEL_PATH, model=model, num_classes=NUM_CLASSES, lr=LR, weights=WEIGHTS)
    return classifier


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


def main():
    seed_all(SEED)
    for model in models_list:
        
        if MODEL_PATH is not None:
            if TEST_ONLY:
                model = load_model(model, mode='test')
            else:
                model = load_model(model, mode='train')
        
        mode = str(TYPE)
        if isinstance(model, Classifier):
            model_type = type(model.model).__name__
            input_size = model.model.input_size
            run_save_dir = mode + '/' + type(model.model).__name__  + '/' + str(DATE_NOW)
        else:
            model_type = type(model).__name__
            input_size = model.input_size
            run_save_dir = mode + '/' + type(model).__name__  + '/' + str(DATE_NOW)
        
        run_save_dir = run_save_dir.replace(" ", "_")
        path = str(CHECKPOINTS_DIR)
        checkpoints_run_dir = path + '/' + run_save_dir
        Path(checkpoints_run_dir).mkdir(mode=777, parents=True, exist_ok=True)
        
        data_module = EyeDiseaseDataModule(
            csv_path='/media/data/adam_chlopowiec/eye_image_classification/pretrain_collected_data_splits.csv',
            train_split_name='train',
            val_split_name='val',
            test_split_name='test',
            train_transforms=train_transforms(input_size, NORMALIZE, cv2.INTER_NEAREST),
            val_transforms=test_val_transforms(input_size, NORMALIZE, cv2.INTER_NEAREST),
            test_transforms=test_val_transforms(input_size, NORMALIZE, cv2.INTER_NEAREST),
            image_path_name='Path',
            target_name='Label',
            split_name='Split',
            batch_size=BATCH_SIZE,
            num_workers=12,
            shuffle_train=True,
            resampler=RESAMPLER
        )
        data_module.prepare_data()

        hparams = {
            'dataset': type(data_module).__name__,
            'model_type': model_type,
            'lr': LR,
            'batch_size': BATCH_SIZE,
            'optimizer': 'Adam',
            'resampler': RESAMPLER.__name__,
            'num_classes': NUM_CLASSES,
            'run_id': run_save_dir
        }

        logger = WandbLogger(
            save_dir=LOGS_DIR,
            config=hparams,
            project=PROJECT_NAME,
            log_model=False,
            entity=ENTITY_NAME
        )

        callbacks = [
            EarlyStopping(
                monitor=MONITOR,
                patience=PATIENCE,
                mode='min'
            ),
            ModelCheckpoint(
                monitor=MONITOR,
                dirpath=checkpoints_run_dir,
                save_top_k=1,
                filename=model_type,
                save_weights_only=True
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
            weights=WEIGHTS,
            test_only=TEST_ONLY
        )
        logger.experiment.finish()


if __name__ == '__main__':
    main()
