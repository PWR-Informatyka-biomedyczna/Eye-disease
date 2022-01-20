import random
import argparse
from pathlib import Path
from methods.classifier import Classifier

import numpy as np
import cv2
import torch
import albumentations as A
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchvision import transforms


from dataset import EyeDiseaseDataModule, resamplers
from dataset.transforms import test_val_transforms, train_transforms, Albument, ToNumpy
#from methods import ResNet18Model, ResNet50Model, EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, Xception
#from methods import DenseNet161, ResNext50, ResNext101, RegNetY3_2gf, VGG16, RegNetX3_2gf, RegNetX800MF, RegNetY800MF, RegNetY8gf
from methods import RegNetY3_2gf
from settings import LOGS_DIR, CHECKPOINTS_DIR, PROJECT_DIR
from training import train_test
from dataset.resamplers import threshold_to_glaucoma_with_ros, binary_thresh_to_20k_equal, identity_resampler


# PL_TORCH_DISTRIBUTED_BACKEND=gloo poetry run python3 -m experiments.base_experiment
# experiment setup
SEED = 0
PROJECT_NAME = 'AugmentationOptimizing'
NUM_CLASSES = 4
LR = 1e-4
OPTIM = 'nadam'
BATCH_SIZE = 128
MAX_EPOCHS = 200
NORMALIZE = True
MONITOR = 'val_loss'
PATIENCE = 5
GPUS = -1
ENTITY_NAME = 'kn-bmi'
#RESAMPLER = threshold_to_glaucoma_with_ros
RESAMPLER = threshold_to_glaucoma_with_ros
TYPE = 'training' # pretraining, training, training-from-pretrained
#MODEL_PATH = '/home/adam_chlopowiec/data/eye_image_classification/Eye-disease/checkpoints/pretraining/ResNet18Model/2021-12-15_15:59:16.045619/ResNet18Model.ckpt'
MODEL_PATH = None
TEST_ONLY = False
PRETRAINING = False
BINARY = False
TRAIN_SPLIT_NAME = 'train'
VAL_SPLIT_NAME = 'val'
TEST_SPLIT_NAME = 'test'

models_list = [
        RegNetY3_2gf(NUM_CLASSES)
    ]


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)

def load_model(model, optimizer=None, lr_scheduler=None, mode: str = 'train', lr=1e-4, weights=torch.Tensor([1, 1, 2.5, 2])):
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
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=4e-3, dampening=0, nesterov=True, weight_decay=1e-6)
    if config == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-7, amsgrad=True)
    if config == 'nadam':
        return torch.optim.NAdam(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-5, 
                                 momentum_decay=4e-3)

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

def init_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rotate_p', type=float)
    parser.add_argument('--horizontal_p', type=float)
    parser.add_argument('--vertical_p', type=float)
    parser.add_argument('--brightness_p', type=float)
    parser.add_argument('--clahe_p', type=float)
    parser.add_argument('--cutout_p', type=float)
    parser.add_argument('--rotate_limit', type=int)
    parser.add_argument('--brightness_limit', type=float)
    parser.add_argument('--contrast_limit', type=float)
    parser.add_argument('--clip_limit', type=float)
    parser.add_argument('--num_holes', type=int)
    parser.add_argument('--max_h_size', type=int)
    parser.add_argument('--max_w_size', type=int)
    return parser


def sweep_train_transforms(target_size, config):
    rotate_limit = (-config.rotate_limit, config.rotate_limit)
    rotate_p = config.rotate_p
    horizontal_p = config.horizontal_p
    vertical_p = config.vertical_p
    brightness_p = config.brightness_p
    brightness_limit = config.brightness_limit
    contrast_limit = config.contrast_limit
    clip_limit = config.clip_limit
    clahe_p = config.clahe_p
    num_holes = config.num_holes
    max_h_size = config.max_h_size
    max_w_size = config.max_w_size
    cutout_p = config.cutout_p

    aug_A = A.Compose(
                    [
                        A.Resize(target_size[0], target_size[1], interpolation=cv2.INTER_NEAREST),
                        A.Rotate(limit=rotate_limit, p=rotate_p, interpolation=cv2.INTER_NEAREST),
                        A.HorizontalFlip(p=horizontal_p),
                        A.VerticalFlip(p=vertical_p),
                        A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=brightness_p),
                        A.CLAHE(clip_limit=clip_limit, p=clahe_p),
                        A.Cutout(num_holes=num_holes, max_h_size=max_h_size, max_w_size=max_w_size, p=cutout_p),
                    ])

    albument = Albument(aug_A)

    transforms_list = [
        ToNumpy(),
        albument,
        transforms.ToTensor()
    ]

    if NORMALIZE:
        transforms_list.append(transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        ))
    return transforms.Compose(transforms_list)


def main():
    seed_all(SEED)
    parser = init_argument_parser()
    config = parser.parse_args()
    weights = torch.Tensor([1, 0.9, 2.5, 1.5])

    for model in models_list:
        suffix = str(LR) + '_' + 'nadam'
        optimizer = init_optimizer(model, OPTIM, lr=LR)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        if MODEL_PATH is not None:
            if TEST_ONLY:
                model = load_model(model, optimizer=optimizer, lr_scheduler=lr_scheduler, mode='test', lr=LR, weights=weights, config=None)
            else:
                model = load_model(model, optimizer=optimizer, lr_scheduler=lr_scheduler, mode='train', lr=LR, weights=weights, config=None)
        
        checkpoints_run_dir, model_type, input_size = create_log_path(model, suffix)
        #Path(checkpoints_run_dir).mkdir(mode=777, parents=True, exist_ok=True)
            
        data_module = EyeDiseaseDataModule(
            csv_path='/home/adamch/data/resized_collected_data_splits.csv',
            train_split_name=TRAIN_SPLIT_NAME,
            val_split_name=VAL_SPLIT_NAME,
            test_split_name=TEST_SPLIT_NAME,
            train_transforms=sweep_train_transforms(input_size, config),
            val_transforms=test_val_transforms(input_size, NORMALIZE, cv2.INTER_NEAREST),
            test_transforms=test_val_transforms(input_size, NORMALIZE, cv2.INTER_NEAREST),
            image_path_name='Path',
            target_name='Label',
            split_name='Split',
            batch_size=BATCH_SIZE,
            num_workers=24,
            shuffle_train=True,
            resampler=RESAMPLER
        )
        data_module.prepare_data()

        hparams = {
                    'dataset': type(data_module).__name__,
                    'model_type': model_type,
                    'lr': LR,
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
            # ModelCheckpoint(
            #     monitor=MONITOR,
            #     dirpath=checkpoints_run_dir,
            #     save_top_k=1,
            #     filename=model_type,
            #     save_weights_only=True
            # )
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
            weights=weights,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            test_only=TEST_ONLY
        )
        logger.experiment.finish()


if __name__ == '__main__':
    main()
