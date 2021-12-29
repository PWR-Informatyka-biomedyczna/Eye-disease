import time
import hashlib
import random
import os
import datetime
import argparse
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
PROJECT_NAME = 'EfficientNetB4Pretraining'
NUM_CLASSES = 2
LR = [1e-4, 1e-3, 1e-2]
OPTIM = ['adamw', 'sgd']
BATCH_SIZE = 64
MAX_EPOCHS = 100
NORMALIZE = True
MONITOR = 'val_loss'
PATIENCE = 5
GPUS = -1
ENTITY_NAME = 'kn-bmi'
RESAMPLER = resamplers.identity_resampler
TYPE = 'pretraining' # pretraining, training, training-from-pretrained
MODEL_PATH = None
TEST_ONLY = False
TRAIN_SPLIT_NAME = 'pretrain'
VAL_SPLIT_NAME = 'preval'
TEST_SPLIT_NAME = 'pretest'

models_list = [
        #EfficientNetB0(NUM_CLASSES),
        #EfficientNetB2(NUM_CLASSES),
        EfficientNetB4(NUM_CLASSES)
        #Xception(NUM_CLASSES),
        #DenseNet(NUM_CLASSES),
        #ResNext50(NUM_CLASSES),
        #ResNet18Model(NUM_CLASSES),
        #ResNet50Model(NUM_CLASSES)
    ]

def load_model(model, optimizer=None, lr_scheduler=None, mode: str = 'train', lr=1e-4, weights=torch.Tensor([1, 1, 2.5, 2]), config=None):
    classifier = Classifier(
                model=model,
                num_classes=NUM_CLASSES,
                lr=lr,
                weights=torch.Tensor([1, 1]),
                optimizer=None,
                lr_scheduler=None
                )
    in_features = model.get_last_layer().in_features
    out_features = model.get_last_layer().out_features
    if mode == 'train':
        if out_features > 2:
            model.set_last_layer(torch.nn.Linear(in_features, 2))
        classifier.load_from_checkpoint(checkpoint_path=MODEL_PATH, model=model, 
                                        num_classes=NUM_CLASSES, lr=lr, weights=torch.Tensor([1, 1]), optimizer=None, lr_scheduler=None)
        classifier.model.set_last_layer(torch.nn.Linear(in_features, NUM_CLASSES))
    elif mode == 'test':
        if out_features < NUM_CLASSES:
            model.set_last_layer(torch.nn.Linear(in_features, NUM_CLASSES))
        classifier.load_from_checkpoint(checkpoint_path=MODEL_PATH, model=model, 
                                        num_classes=NUM_CLASSES, lr=lr, weights=torch.Tensor(1, 1, 1, 1), 
                                        optimizer=None, lr_scheduler=None)
    classifier.optimizer = optimizer
    classifier.lr_scheduler = lr_scheduler
    classifier.criterion.weight = weights
    
    return classifier


def init_const_values(config):
    lr = config.learning_rate
    batch_size = config.batch_size
    weight_0 = config.weight_0
    weight_1 = config.weight_1
    weight_2 = config.weight_2
    weight_3 = config.weight_3
    weights = torch.Tensor([weight_0, weight_1, weight_2, weight_3])
    return lr, batch_size, weights


def init_optimizer(model, config, lr=1e-4):
    if MODEL_PATH is not None:
        classifier = Classifier(
                    model=model,
                    num_classes=NUM_CLASSES,
                    lr=lr,
                    weights=torch.Tensor([1, 1]),
                    optimizer=None,
                    lr_scheduler=None
                    )
        in_features = model.get_last_layer().in_features
        out_features = model.get_last_layer().out_features
        if out_features > 2:
                model.set_last_layer(torch.nn.Linear(in_features, 2))
        dummy_classifier = classifier.load_from_checkpoint(checkpoint_path=MODEL_PATH, model=model, 
                                            num_classes=NUM_CLASSES, lr=1e-4, weights=torch.Tensor([1, 1]), optimizer=None, lr_scheduler=None)
        dummy_classifier.model.set_last_layer(torch.nn.Linear(in_features, NUM_CLASSES))
        model = dummy_classifier.model

    if config == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=4e-3, dampening=0, nesterov=True, weight_decay=1e-5)
    if config == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5, amsgrad=True)

    optimizer = config.optimizer
    if optimizer == 'adam':
        beta = config.beta
        weight_decay = config.weight_decay
        amsgrad = config.amsgrad
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, beta), weight_decay=weight_decay, amsgrad=amsgrad)
    
    elif optimizer == 'adamw':
        beta = config.beta
        weight_decay = config.weight_decay
        amsgrad = config.amsgrad    
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, beta), weight_decay=weight_decay, amsgrad=amsgrad)
    
    elif optimizer == 'adamax':
        beta = config.beta
        weight_decay = config.weight_decay
        return torch.optim.Adamax(model.parameters(), lr=lr, betas=(0.9, beta), weight_decay=weight_decay)
    
    elif optimizer == 'radam':
        beta = config.beta
        weight_decay = config.weight_decay
        return torch.optim.RAdam(model.parameters(), lr=lr, betas=(0.9, beta), weight_decay=weight_decay)
    
    elif optimizer == 'nadam':
        beta = config.beta
        momentum_decay = config.momentum_decay
        weight_decay = config.weight_decay
        return torch.optim.NAdam(model.parameters(), lr=lr, betas=(0.9, beta), weight_decay=weight_decay, momentum_decay=momentum_decay)
    
    elif optimizer == 'rmsprop':
        weight_decay = config.weight_decay
        alpha = config.alpha
        momentum = config.momentum 
        centered = config.centered
        return torch.optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, momentum=momentum, weight_decay=weight_decay, centered=centered)
    
    elif optimizer == 'sgd':
        weight_decay = config.weight_decay
        dampening = config.dampening
        nesterov = config.nesterov
        momentum = config.momentum
        if nesterov:
            return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                               momentum=momentum, dampening=0, nesterov=nesterov)
        else:
            return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                               momentum=momentum, dampening=dampening, nesterov=nesterov)
    
    elif optimizer == 'asgd':
        alpha = config.alpha_asgd
        lambda_ = config.lambda_asgd
        t0 = config.t0_asgd
        weight_decay = config.weight_decay
        return torch.optim.ASGD(model.parameters(), lr=lr, lambd=lambda_, alpha=alpha, t0=t0, weight_decay=weight_decay)
    
    return None


def init_scheduler(optimizer, config):
    scheduler_params = config.lr_scheduler
    scheduler_params = scheduler_params.split('_')
    if scheduler_params[0] == 'multiplicativelr':
        lr_lambda = lambda epoch: float(scheduler_params[1])
        return torch.optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer, lr_lambda=lr_lambda)
    elif scheduler_params[0] == 'cosinelr':
        t_max = int(scheduler_params[1])
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=t_max)
    elif scheduler_params[0] == 'cosinewarmlr':
        t_0 = int(scheduler_params[1])
        t_mul = int(scheduler_params[2])
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=t_0, T_mult=t_mul)
    elif scheduler_params[0] == 'None':
        return None
    return None

def init_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--weight_0', type=float)
    parser.add_argument('--weight_1', type=float)
    parser.add_argument('--weight_2', type=float)
    parser.add_argument('--weight_3', type=float)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--amsgrad', type=bool)
    parser.add_argument('--lr_scheduler', type=str)
    parser.add_argument('--lr_lambda', type=float)
    parser.add_argument('--t_max', type=int)
    parser.add_argument('--t_0', type=int)
    parser.add_argument('--t_mul', type=int)
    parser.add_argument('--momentum_decay', type=float)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--centered', type=bool)
    parser.add_argument('--dampening', type=float)
    parser.add_argument('--nesterov', type=bool)
    return parser


def create_log_path(model, suffix):
    mode = str(TYPE)
    if isinstance(model, Classifier):
        model_type = type(model.model).__name__
        input_size = model.model.input_size
        run_save_dir = mode + '/' + type(model.model).__name__  + '/' + suffix
    else:
        model_type = type(model).__name__
        input_size = model.input_size
        run_save_dir = mode + '/' + type(model).__name__  + '/' + suffix
    
    run_save_dir = run_save_dir.replace(" ", "_")
    path = str(CHECKPOINTS_DIR)
    checkpoints_run_dir = path + '/' + run_save_dir
    return checkpoints_run_dir, model_type, input_size


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


def main():
    seed_all(SEED)
    weights = torch.Tensor([1, 2])
    for optim in OPTIM:
        for lr in LR:
            for model in models_list:
                suffix = str(lr) + '_' + optim
                optimizer = init_optimizer(model, optim, lr=lr)
                lr_scheduler = None
                if MODEL_PATH is not None:
                    if TEST_ONLY:
                        model = load_model(model, optimizer=optimizer, lr_scheduler=lr_scheduler, mode='test', lr=lr, weights=weights, config=None)
                    else:
                        model = load_model(model, optimizer=optimizer, lr_scheduler=lr_scheduler, mode='train', lr=lr, weights=weights, config=None)
                
                checkpoints_run_dir, model_type, input_size = create_log_path(model, suffix)
                Path(checkpoints_run_dir).mkdir(mode=777, parents=True, exist_ok=True)
                
                data_module = EyeDiseaseDataModule(
                    csv_path='../pretrain_collected_data_splits.csv',
                    train_split_name=TRAIN_SPLIT_NAME,
                    val_split_name=VAL_SPLIT_NAME,
                    test_split_name=TEST_SPLIT_NAME,
                    train_transforms=train_transforms(input_size, NORMALIZE, cv2.INTER_NEAREST),
                    val_transforms=test_val_transforms(input_size, NORMALIZE, cv2.INTER_NEAREST),
                    test_transforms=test_val_transforms(input_size, NORMALIZE, cv2.INTER_NEAREST),
                    image_path_name='Path',
                    target_name='Label',
                    split_name='Split',
                    batch_size=BATCH_SIZE,
                    num_workers=-1,
                    shuffle_train=True,
                    resampler=RESAMPLER
                )
                data_module.prepare_data()

                hparams = {
                    'dataset': type(data_module).__name__,
                    'model_type': model_type,
                    'lr': lr,
                    'batch_size': BATCH_SIZE,
                    'optimizer': type(optimizer).__name__,
                    'resampler': RESAMPLER.__name__,
                    'num_classes': NUM_CLASSES,
                    #'run_id': run_save_dir
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
                    lr=lr,
                    callbacks=callbacks,
                    logger=logger,
                    weights=weights,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    test_only=TEST_ONLY,
                    precision=16
                )
                logger.experiment.finish()


if __name__ == '__main__':
    main()
