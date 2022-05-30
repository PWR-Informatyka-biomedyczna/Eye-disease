from cgi import test
import hashlib
import time
import copy
from pathlib import Path

import argparse

import cv2
import torch
from torch import nn
import timm
import pandas as pd
from timm.scheduler.cosine_lr import CosineLRScheduler
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from adamp import AdamP


from experiments.common import seed_all, freeze, unfreeze, get_train_params_count, load_lightning_model, layer_decay
from utils.callbacks import EMA
from dataset import EyeDiseaseDataModule
from dataset.transforms import test_val_transforms, train_transforms
from methods import RegNetY3_2gf, ResNet50Model, ConvNextTiny
from settings import LOGS_DIR, CHECKPOINTS_DIR, PROJECT_DIR
from training import train_test
from dataset.resamplers import identity_resampler

MODELS_DICT = {
    'resnet': ResNet50Model,
    'regnet': RegNetY3_2gf,
    'convnext': ConvNextTiny
}

SEED = 0
PROJECT_NAME = 'EyeDiseaseFinalCV'
NUM_CLASSES = 2
LR = 1e-4
OPTIM = 'adamp'
BATCH_SIZE = 32
MAX_EPOCHS = 100
NORMALIZE = True
MONITOR = 'val_loss'
PATIENCE = 5
GPUS = -1
ENTITY_NAME = 'kn-bmi'
RESAMPLER = identity_resampler
TEST_ONLY = False
PRETRAINING = True
BINARY = False
TRAIN_SPLIT_NAME = 'pretrain'
VAL_SPLIT_NAME = 'val'
TEST_SPLIT_NAME = 'test'
#MODEL_PATH = '/home/adam_chlopowiec/data/eye_image_classification/Eye-disease/checkpoints/pretraining/RegNetY3_2gf/0.0001_radam/RegNetY3_2gf-v1.ckpt'
MODEL_PATH = None
cross_val = False
cross_val_test = False
k_folds = 10
n_runs = 5
LABEL_SMOOTHING = 0.1
LAYER_DECAY = 0.4
EMA_DECAY = 0.0
t_initial = 30
eta_min = 1e-7
warmup_lr_init = 1e-6
warmup_epochs = 5
cycle_mul = 1
cycle_decay = 0.1

PRETRAIN = pd.read_csv('csvs/pretrain.csv')
FINETUNE = pd.read_csv('csvs/finetune_splits.csv')

def init_optimizer(params, optimizer, lr=1e-4):
    if optimizer == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=4e-3, dampening=0, nesterov=True, weight_decay=1e-6)
    if optimizer == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-3, amsgrad=True)
    if optimizer == 'nadam':
        return torch.optim.NAdam(params, lr=lr, betas=(0.9, 0.999), weight_decay=1e-5,
                                 momentum_decay=4e-3)
    if optimizer == 'radam':
        return torch.optim.RAdam(params, lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)
    if optimizer == 'adamp':
        return AdamP(params, lr=lr, betas=(0.9, 0.999), weight_decay=1e-2, nesterov=True)
    
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', dest='model', type=str, required=True)
    return parser.parse_args()


def pre_train():
    model_cls = parse_args().model
    model = MODELS_DICT[model_cls](2)
    pre_train_weights = torch.Tensor([0.647, 1])

    # optimization
    parameters = layer_decay(model, LR, LAYER_DECAY)
    optimizer = init_optimizer(parameters, OPTIM, lr=LR)
    lr_scheduler = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=t_initial,
        lr_min=eta_min,
        warmup_lr_init=warmup_lr_init,
        warmup_t=warmup_epochs,
        cycle_mul=cycle_mul,
        cycle_decay=cycle_decay,
    )

    print(PRETRAIN['Split'].unique())

    # pre-training
    data_module = EyeDiseaseDataModule(
        csv_path='/media/data/adam_chlopowiec/eye_image_classification/pretrain_corrected_data_splits.csv',
        train_split_name=TRAIN_SPLIT_NAME,
        val_split_name=VAL_SPLIT_NAME,
        test_split_name=TEST_SPLIT_NAME,
        train_transforms=train_transforms(model.input_size, NORMALIZE, cv2.INTER_NEAREST),
        val_transforms=test_val_transforms(model.input_size, NORMALIZE, cv2.INTER_NEAREST),
        test_transforms=test_val_transforms(model.input_size, NORMALIZE, cv2.INTER_NEAREST),
        image_path_name='Path',
        target_name='Label',
        split_name='Split',
        batch_size=BATCH_SIZE,
        num_workers=12,
        shuffle_train=True,
        resampler=RESAMPLER,
        pretraining=PRETRAINING,
        binary=BINARY,
        df=PRETRAIN
    )
    data_module.prepare_data()

    hparams = {
            'dataset': type(data_module).__name__,
            'model_type': type(model).__name__,
            'lr': LR,
            'batch_size': BATCH_SIZE,
            'optimizer': type(optimizer).__name__,
            'resampler': RESAMPLER.__name__,
            'num_classes': NUM_CLASSES,
            'pre-train': True
        }

    logger = WandbLogger(
        save_dir=LOGS_DIR,
        config=hparams,
        project=PROJECT_NAME,
        log_model=False,
        entity=ENTITY_NAME
    )

    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            dirpath='models',
            save_top_k=1,
            filename=type(model).__name__,
            save_weights_only=True
        ),
        EarlyStopping(
            monitor=MONITOR,
            patience=PATIENCE,
            mode='min'
        )
    ]
    if EMA_DECAY > 0:
        callbacks.append(
            EMA(EMA_DECAY)
        )

    best_model = train_test(
        model=model,
        datamodule=data_module,
        max_epochs=MAX_EPOCHS,
        num_classes=NUM_CLASSES,
        gpus=GPUS,
        lr=LR,
        callbacks=callbacks,
        logger=logger,
        weights=pre_train_weights,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        label_smoothing=LABEL_SMOOTHING,
        test=False
    )
    print(best_model)
    return best_model


def get_fold(df, test, val):
    df_test = df[df['Fold'] == test]
    df_val = df[df['Fold'] == val]
    df_train = df[~df['Fold'].isin([test, val])]

    df_test['Split'] = 'test'
    df_val['Split'] = 'val'
    df_train['Split'] = 'train'
    new_df = pd.concat([df_train, df_val, df_test])
    assert len(df_test) + len(df_train) + len(df_val) == len(df) == len(new_df), 'Kurwa xD'
    return new_df



def fine_tune(model, test_fold=0):

    # wchujwazne
    ll = model.get_last_layer()
    model.set_last_layer(nn.Linear(
        ll.in_features,
        4
    ))
    # koniec


    val_fold = (test_fold + 1) % 10

    df = get_fold(FINETUNE, test=test_fold, val=val_fold)
    

    weights = torch.Tensor([0.408, 0.408, 1, 0.408])

    # optimization
    parameters = layer_decay(model, LR, LAYER_DECAY)
    optimizer = init_optimizer(parameters, OPTIM, lr=LR)
    lr_scheduler = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=t_initial,
        lr_min=eta_min,
        warmup_lr_init=warmup_lr_init,
        warmup_t=warmup_epochs,
        cycle_mul=cycle_mul,
        cycle_decay=cycle_decay,
    )

    # raise Exception(df['Label'].unique())

    # pre-training
    data_module = EyeDiseaseDataModule(
        csv_path='/media/data/adam_chlopowiec/eye_image_classification/pretrain_corrected_data_splits.csv',
        train_split_name=TRAIN_SPLIT_NAME,
        val_split_name=VAL_SPLIT_NAME,
        test_split_name=TEST_SPLIT_NAME,
        train_transforms=train_transforms(model.input_size, NORMALIZE, cv2.INTER_NEAREST),
        val_transforms=test_val_transforms(model.input_size, NORMALIZE, cv2.INTER_NEAREST),
        test_transforms=test_val_transforms(model.input_size, NORMALIZE, cv2.INTER_NEAREST),
        image_path_name='Path',
        target_name='Label',
        split_name='Split',
        batch_size=BATCH_SIZE,
        num_workers=12,
        shuffle_train=True,
        resampler=RESAMPLER,
        pretraining=PRETRAINING,
        binary=BINARY,
        df=df
    )
    data_module.prepare_data()

    hparams = {
            'dataset': type(data_module).__name__,
            'model_type': type(model).__name__,
            'lr': LR,
            'batch_size': BATCH_SIZE,
            'optimizer': type(optimizer).__name__,
            'resampler': RESAMPLER.__name__,
            'num_classes': NUM_CLASSES,
            'pre-train': False,
            'test-fold-num': test_fold
        }

    logger = WandbLogger(
        save_dir=LOGS_DIR,
        config=hparams,
        project=PROJECT_NAME,
        log_model=False,
        entity=ENTITY_NAME
    )

    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            dirpath='models',
            save_top_k=1,
            filename=type(model).__name__,
            save_weights_only=True
        ),
        EarlyStopping(
            monitor=MONITOR,
            patience=PATIENCE,
            mode='min'
        )
    ]
    if EMA_DECAY > 0:
        callbacks.append(
            EMA(EMA_DECAY)
        )

    train_test(
        model=model,
        datamodule=data_module,
        max_epochs=MAX_EPOCHS,
        num_classes=4,
        gpus=GPUS,
        lr=LR,
        callbacks=callbacks,
        logger=logger,
        weights=weights,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        label_smoothing=LABEL_SMOOTHING,
        test=True
    )



def main():
    seed_all(0)
    best_model = pre_train()
    # fine-tuning
    for i in range(10):
        fine_tune(copy.deepcopy(best_model), test_fold=i)

    
if __name__ == '__main__':
    main()
