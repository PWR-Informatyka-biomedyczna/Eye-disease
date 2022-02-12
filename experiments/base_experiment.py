import hashlib
import time
from pathlib import Path
from methods.classifier import Classifier

import numpy as np
import cv2
import torch
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
PROJECT_NAME = 'EyeDiseaseExperiments'
NUM_CLASSES = 2
LR = 1e-4
OPTIM = 'nadam'
BATCH_SIZE = 24
MAX_EPOCHS = 200
NORMALIZE = True
MONITOR = 'val_loss'
PATIENCE = 5
GPUS = -1
ENTITY_NAME = 'kn-bmi'
RESAMPLER = identity_resampler
TEST_ONLY = False
PRETRAINING = False
BINARY = True
TRAIN_SPLIT_NAME = 'train'
VAL_SPLIT_NAME = 'val'
TEST_SPLIT_NAME = 'test'

model = RegNetY3_2gf(NUM_CLASSES)


def init_optimizer(model, optimizer, lr=1e-4):
    if optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=4e-3, dampening=0, nesterov=True, weight_decay=1e-6)
    if optimizer == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-7, amsgrad=True)
    if optimizer == 'nadam':
        return torch.optim.NAdam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5, 
                                 momentum_decay=4e-3)
    if optimizer == 'radam':
        return torch.optim.RAdam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)
    
    return None


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
        image_path_name='Path',
        target_name='Label',
        split_name='Split',
        batch_size=BATCH_SIZE,
        num_workers=12,
        shuffle_train=True,
        resampler=RESAMPLER,
        pretraining=PRETRAINING,
        binary=BINARY
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
            monitor="val_sensitivity_class_1",
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
        test_only=TEST_ONLY
    )
    logger.experiment.finish()


if __name__ == '__main__':
    main()
