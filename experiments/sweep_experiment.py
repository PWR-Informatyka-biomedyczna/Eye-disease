import time
import hashlib
import random
import os

import numpy as np
import cv2
import torch
from torch.optim import lr_scheduler
import torchvision
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint



from dataset import EyeDiseaseDataModule, resamplers
from dataset.transforms import test_val_transforms, train_transforms
from methods import ResNet18Model, ResNet50Model, EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, Xception
from methods import DenseNet, ResNext50, ResNext101
from settings import LOGS_DIR, CHECKPOINTS_DIR
from training import train_test


# experiment setup
SEED = 0
PROJECT_NAME = 'ResNet18Optimizing'
NUM_CLASSES = 4
lr = 1e-4
batch_size = 12
MAX_EPOCHS = 100
NORMALIZE = True
MONITOR = 'val_loss'
PATIENCE = 5
GPUS = -1
ENTITY_NAME = 'kn-bmi'
RESAMPLER = resamplers.identity_resampler
weights = torch.Tensor([1, 2, 2.5, 1.5])

models_list = [
        #EfficientNetB0(NUM_CLASSES),
        #EfficientNetB2(NUM_CLASSES),
        #Xception(NUM_CLASSES),
        #DenseNet(NUM_CLASSES),
        #ResNext50(NUM_CLASSES),
        ResNet18Model(NUM_CLASSES),
        #ResNet50Model(NUM_CLASSES)
    ]


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


def init_const_values(config):
    lr = config['learning_rate']
    batch_size = config['batch_size']
    weight_0 = config['weight_0']
    weight_1 = config['weight_1']
    weight_2 = config['weight_2']
    weight_3 = config['weight_3']
    weights = [weight_0, weight_1, weight_2, weight_3]
    return lr, batch_size, weights


def init_optimizer(model, config):
    optimizer = config['optimizer']
    if optimizer == 'adam':
        beta = config['beta']
        weight_decay = config['weight_decay']
        amsgrad = config['amsgrad']
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, beta), weight_decay=weight_decay, amsgrad=amsgrad)
    elif optimizer == 'adamw':
        beta = config['beta']
        weight_decay = config['weight_decay']
        amsgrad = config['amsgrad']
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, beta), weight_decay=weight_decay, amsgrad=amsgrad)
    elif optimizer == 'adamax':
        beta = config['beta']
        weight_decay = config['weight_decay']
        return torch.optim.Adamax(model.parameters(), lr=lr, betas=(0.9, beta), weight_decay=weight_decay)
    return None


def init_scheduler(optimizer, config):
    scheduler = config['lr_scheduler']
    if scheduler == 'multiplicative_lr':
        lr_lambda = lambda epoch: config['lr_lambda']
        return torch.optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer, lr_lambda=lr_lambda)
    elif scheduler == 'cosine_lr':
        t_max = config['t_max']
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=t_max)
    elif scheduler == 'cosine_warm_lr':
        t_max = config['t_max']
        t_0 = config['t_0']
        t_mul = config['t_mul']
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=t_0, T_mult=t_mul)
    return None

def main():
    COUNTER = 0
    seed_all(SEED)
    config = wandb.config
    lr, batch_size, weights = init_const_values(config)
    
    for model in models_list:
        run_id = hashlib.md5(
            bytes(str(time.time()), encoding='utf-8')
        ).hexdigest()
        optimizer = init_optimizer(model, config)
        lr_scheduler = init_scheduler(optimizer, config)
        checkpoints_run_dir = CHECKPOINTS_DIR / run_id
        COUNTER += 1
        print(run_id, 'ZROBIONO', time.time(), 'COUNTER ', COUNTER)
        os.mkdir(checkpoints_run_dir)
        data_module = EyeDiseaseDataModule(
            csv_path='/media/data/adam_chlopowiec/eye_image_classification/resized_collected_data_splits.csv',
            train_split_name='train',
            val_split_name='val',
            test_split_name='test',
            train_transforms=train_transforms(model.input_size, NORMALIZE, cv2.INTER_NEAREST),
            val_transforms=test_val_transforms(model.input_size, NORMALIZE, cv2.INTER_NEAREST),
            test_transforms=test_val_transforms(model.input_size, NORMALIZE, cv2.INTER_NEAREST),
            image_path_name='Path',
            target_name='Label',
            split_name='Split',
            batch_size=batch_size,
            num_workers=12,
            shuffle_train=True,
            resampler=RESAMPLER
        )
        data_module.prepare_data()

        hparams = {
            'dataset': type(data_module).__name__,
            'model_type': type(model).__name__,
            'lr': lr,
            'batch_size': batch_size,
            'optimizer': type(optimizer).__name__,
            'resampler': RESAMPLER.__name__,
            'num_classes': NUM_CLASSES,
            'run_id': run_id
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
            )
        ]
        train_test(
            model=model,
            datamodule=data_module,
            max_epochs=MAX_EPOCHS,
            num_classes=NUM_CLASSES,
            gpus=GPUS,
            lr=lr,
            callbacks=callbacks,
            logger=logger,
            weights=weights,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler
        )
        logger.experiment.finish()


if __name__ == '__main__':
    main()
