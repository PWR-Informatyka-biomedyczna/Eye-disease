import hashlib
import time
from pathlib import Path
from methods.classifier import Classifier

import wandb
import numpy as np
import cv2
import torch
import torch.nn as nn
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


from experiments.common import seed_all
from dataset import EyeDiseaseDataModule, resamplers
from dataset.transforms import test_val_transforms, train_transforms
from methods import ResNet18Model, ResNet50Model, EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, Xception
from methods import DenseNet161, ResNext50, ResNext101, RegNetY3_2gf, VGG16, RegNetX3_2gf, RegNetX800MF, RegNetY800MF, RegNetY8gf
from settings import LOGS_DIR, CHECKPOINTS_DIR, PROJECT_DIR
from training import train_test
from dataset.resamplers import threshold_to_glaucoma_with_ros, binary_thresh_to_20k_equal, identity_resampler

SEED = 0
<<<<<<< HEAD
PROJECT_NAME = 'EyeDiseaseExperiments'
NUM_CLASSES = 2
LR = 1e-4
OPTIM = 'nadam'
BATCH_SIZE = 24
MAX_EPOCHS = 200
=======
PROJECT_NAME = 'TransferLearningWithPretraining'
num_classes = 2
LR = 1e-4
OPTIM = 'radam'
BATCH_SIZE = 128
MAX_EPOCHS = 100
>>>>>>> kaggle-version
NORMALIZE = True
MONITOR = 'val_loss'
PATIENCE = 5
GPUS = -1
ENTITY_NAME = 'kn-bmi'
RESAMPLER = identity_resampler
<<<<<<< HEAD
TEST_ONLY = False
PRETRAINING = False
BINARY = True
TRAIN_SPLIT_NAME = 'train'
VAL_SPLIT_NAME = 'val'
TEST_SPLIT_NAME = 'test'

model = RegNetY3_2gf(NUM_CLASSES)


def init_optimizer(model, optimizer, lr=1e-4):
    if optimizer == 'sgd':
=======
TYPE = 'pretraining' # pretraining, training, training-from-pretrained
MODEL_PATH = None
TEST_ONLY = False
pretraining = True
train_split_name = 'pretrain'
val_split_name = 'preval'
test_split_name = 'pretest'
weights = torch.Tensor([1, 2])

model = RegNetY3_2gf(num_classes)


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model(model, optimizer=None, lr_scheduler=None, mode: str = 'train', lr=1e-4, weights=torch.Tensor([1, 1, 2.5, 2])):
    classifier = Classifier(
                model=model,
                num_classes=num_classes,
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
                                        num_classes=num_classes, lr=lr, weights=torch.Tensor([1, 1]), optimizer=None, lr_scheduler=None)
        classifier.model.set_last_layer(torch.nn.Linear(in_features, num_classes))
    elif mode == 'test':
        if out_features < num_classes:
            model.set_last_layer(torch.nn.Linear(in_features, num_classes))
        classifier.load_from_checkpoint(checkpoint_path=MODEL_PATH, model=model, 
                                        num_classes=num_classes, lr=lr, weights=torch.Tensor(1, 1, 1, 1), 
                                        optimizer=None, lr_scheduler=None)
    classifier.optimizer = optimizer
    classifier.lr_scheduler = lr_scheduler
    classifier.criterion.weight = weights
    
    return classifier


def init_optimizer(model, config, lr=1e-4):
    if MODEL_PATH is not None:
        classifier = Classifier(
                    model=model,
                    num_classes=num_classes,
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
                                            num_classes=num_classes, lr=1e-4, weights=torch.Tensor([1, 1]), optimizer=None, lr_scheduler=None)
        dummy_classifier.model.set_last_layer(torch.nn.Linear(in_features, num_classes))
        model = dummy_classifier.model

    if config == 'sgd':
>>>>>>> kaggle-version
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=4e-3, dampening=0, nesterov=True, weight_decay=1e-6)
    if optimizer == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-7, amsgrad=True)
    if optimizer == 'nadam':
        return torch.optim.NAdam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5, 
                                 momentum_decay=4e-3)
<<<<<<< HEAD
    if optimizer == 'radam':
        return torch.optim.RAdam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)
=======
    if config == 'radam':
        return torch.optim.RAdam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)

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
>>>>>>> kaggle-version
    
    return None


<<<<<<< HEAD
def main():
    seed_all(SEED)
    #weights = torch.Tensor([1, 0.9, 1.5, 1.2])
    weights = torch.Tensor([1, 2])
    optimizer = init_optimizer(model, OPTIM, lr=LR)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    run_id = hashlib.md5(
            bytes(str(time.time()), encoding='utf-8')
        ).hexdigest()
    checkpoints_run_dir = CHECKPOINTS_DIR / run_id
    
    Path(checkpoints_run_dir).mkdir(mode=777, parents=True, exist_ok=True)
    
    data_module = EyeDiseaseDataModule(
        csv_path='/media/data/adam_chlopowiec/eye_image_classification/resized_collected_data_splits.csv',
        train_split_name=TRAIN_SPLIT_NAME,
        val_split_name=VAL_SPLIT_NAME,
        test_split_name=TEST_SPLIT_NAME,
        train_transforms=train_transforms(model.input_size, NORMALIZE, cv2.INTER_NEAREST),
        val_transforms=test_val_transforms(model.input_size, NORMALIZE, cv2.INTER_NEAREST),
        test_transforms=test_val_transforms(model.input_size, NORMALIZE, cv2.INTER_NEAREST),
=======
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


def create_log_path(model, suffix):
    mode = TYPE
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


def freeze(model, n_params):
    i = 0
    for param in model.parameters():
        if i < n_params:
            param.requires_grad = False
            i += 1
    
    
def get_train_params_count(model):
    i = 0
    for _ in model.parameters():
        i += 1
    return i


def unfreeze(model, n_params):
    i = 0
    n_params = (int)(n_params)
    n = get_train_params_count(model)
    start = n - n_params
    for param in model.parameters():
        if start <= i and i <= n:
            param.requires_grad = True
        i += 1

def main(model, optimizer, lr_scheduler, initial_lr, max_epoch, weights, save_last, early_stop: bool, log: bool, train_split, val_split, test_split, pretrain, 
         num_classes, test_only=False):
    wandb.finish()
    seed_all(SEED)
    suffix = str(initial_lr) + '_' + OPTIM
    checkpoints_run_dir, model_type, input_size = create_log_path(model, suffix)
    Path(checkpoints_run_dir).mkdir(mode=777, parents=True, exist_ok=True)

    data_module = EyeDiseaseDataModule(
        csv_path='/media/data/adam_chlopowiec/eye_image_classification/pretrain_collected_data_splits.csv',
        train_split_name=train_split,
        val_split_name=val_split,
        test_split_name=test_split,
        train_transforms=train_transforms(input_size, NORMALIZE, cv2.INTER_NEAREST),
        val_transforms=test_val_transforms(input_size, NORMALIZE, cv2.INTER_NEAREST),
        test_transforms=test_val_transforms(input_size, NORMALIZE, cv2.INTER_NEAREST),
>>>>>>> kaggle-version
        image_path_name='Path',
        target_name='Label',
        split_name='Split',
        batch_size=BATCH_SIZE,
<<<<<<< HEAD
        num_workers=12,
        shuffle_train=True,
        resampler=RESAMPLER,
        pretraining=PRETRAINING,
        binary=BINARY
=======
        num_workers=2,
        shuffle_train=True,
        resampler=RESAMPLER,
        pretraining=pretrain,
        #binary=BINARY
>>>>>>> kaggle-version
    )
    data_module.prepare_data()

    hparams = {
        'dataset': type(data_module).__name__,
<<<<<<< HEAD
        'model_type': type(model).__name__,
        'lr': LR,
        'batch_size': BATCH_SIZE,
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
        ),
        ModelCheckpoint(
            monitor="val_loss",
            dirpath=checkpoints_run_dir,
            save_top_k=1,
            filename=type(model).__name__,
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
=======
        'model_type': model_type,
        'lr': initial_lr,
        'batch_size': BATCH_SIZE,
        'optimizer': type(optimizer).__name__,
        'resampler': RESAMPLER.__name__,
        'num_classes': num_classes,
        #'run_id': run_save_dir
    }
    if log:
        logger = WandbLogger(
            save_dir=LOGS_DIR,
            config=hparams,
            project=PROJECT_NAME,
            log_model=False,
            entity=ENTITY_NAME
        )
    else:
        logger = None
    callbacks = []
    if save_last:
        callbacks.append(ModelCheckpoint(
                        monitor=MONITOR,
                        dirpath=checkpoints_run_dir,
                        save_last = True,
                        #save_top_k=1,
                        filename=model_type,
                        save_weights_only=True
                        ))
    else:
        callbacks.append(ModelCheckpoint(
                        monitor=MONITOR,
                        dirpath=checkpoints_run_dir,
                        #save_last = True,
                        save_top_k=1,
                        filename=model_type,
                        save_weights_only=True
                        ))
    if early_stop:
        callbacks.append(EarlyStopping(
                        monitor=MONITOR,
                        patience=PATIENCE,
                        mode='min'
                        ))
    
    train_test(
        model=model,
        datamodule=data_module,
        max_epochs=max_epoch,
        num_classes=num_classes,
        gpus=GPUS,
        lr=initial_lr,
>>>>>>> kaggle-version
        callbacks=callbacks,
        logger=logger,
        weights=weights,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
<<<<<<< HEAD
        test_only=TEST_ONLY
    )
    logger.experiment.finish()
=======
        test_only=test_only,
        precision=32
    )
    
    if log:
        logger.experiment.finish()


# def main():
#     seed_all(SEED)
#     #weights = torch.Tensor([1, 0.9, 1.5, 1.2])
#     weights = torch.Tensor([1, 2])
#     for optim in OPTIM:
#         for lr in LR:
#             for model in models_list:
#                 suffix = str(lr) + '_' + optim
#                 optimizer = init_optimizer(model, optim, lr=lr)
#                 lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
#                 if MODEL_PATH is not None:
#                     if TEST_ONLY:
#                         model = load_model(model, optimizer=optimizer, lr_scheduler=lr_scheduler, mode='test', lr=lr, weights=weights, config=None)
#                     else:
#                         model = load_model(model, optimizer=optimizer, lr_scheduler=lr_scheduler, mode='train', lr=lr, weights=weights, config=None)
                
#                 checkpoints_run_dir, model_type, input_size = create_log_path(model, suffix)
#                 Path(checkpoints_run_dir).mkdir(mode=777, parents=True, exist_ok=True)
                
#                 data_module = EyeDiseaseDataModule(
#                     csv_path='/media/data/adam_chlopowiec/eye_image_classification/resized_collected_data_splits.csv',
#                     train_split_name=TRAIN_SPLIT_NAME,
#                     val_split_name=VAL_SPLIT_NAME,
#                     test_split_name=TEST_SPLIT_NAME,
#                     train_transforms=train_transforms(input_size, NORMALIZE, cv2.INTER_NEAREST),
#                     val_transforms=test_val_transforms(input_size, NORMALIZE, cv2.INTER_NEAREST),
#                     test_transforms=test_val_transforms(input_size, NORMALIZE, cv2.INTER_NEAREST),
#                     image_path_name='Path',
#                     target_name='Label',
#                     split_name='Split',
#                     batch_size=BATCH_SIZE,
#                     num_workers=12,
#                     shuffle_train=True,
#                     resampler=RESAMPLER,
#                     pretraining=PRETRAINING,
#                     binary=BINARY
#                 )
#                 data_module.prepare_data()

#                 hparams = {
#                     'dataset': type(data_module).__name__,
#                     'model_type': model_type,
#                     'lr': lr,
#                     'batch_size': BATCH_SIZE,
#                     'optimizer': type(optimizer).__name__,
#                     'resampler': RESAMPLER.__name__,
#                     'num_classes': NUM_CLASSES,
#                     #'run_id': run_save_dir
#                 }

#                 logger = WandbLogger(
#                     save_dir=LOGS_DIR,
#                     config=hparams,
#                     project=PROJECT_NAME,
#                     log_model=False,
#                     entity=ENTITY_NAME
#                 )

#                 callbacks = [
#                     EarlyStopping(
#                         monitor=MONITOR,
#                         patience=PATIENCE,
#                         mode='min'
#                     ),
#                     ModelCheckpoint(
#                         monitor="val_sensitivity_class_1",
#                         dirpath=checkpoints_run_dir,
#                         save_top_k=1,
#                         filename=model_type,
#                         save_weights_only=True
#                     )
#                 ]
#                 train_test(
#                     model=model,
#                     datamodule=data_module,
#                     max_epochs=MAX_EPOCHS,
#                     num_classes=NUM_CLASSES,
#                     gpus=GPUS,
#                     lr=lr,
#                     callbacks=callbacks,
#                     logger=logger,
#                     weights=weights,
#                     optimizer=optimizer,
#                     lr_scheduler=lr_scheduler,
#                     test_only=TEST_ONLY
#                 )
#                 logger.experiment.finish()
>>>>>>> kaggle-version


if __name__ == '__main__':
    # Fine-tune top layers
    unfreeze(model, get_train_params_count(model))
    freeze(model, get_train_params_count(model) * (1/2))
    optimizer = init_optimizer(model, OPTIM, lr=LR)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    main(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, initial_lr=LR, max_epoch=MAX_EPOCHS, weights=weights, save_last=False, early_stop=True, log=True, 
        train_split=train_split_name, val_split=val_split_name, test_split=test_split_name, pretrain=pretraining, num_classes=num_classes)
