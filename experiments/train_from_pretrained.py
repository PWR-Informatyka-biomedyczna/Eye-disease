import time
import hashlib
import random
import os

import numpy as np
import cv2
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint



from dataset import EyeDiseaseDataModule, resamplers
from dataset.transforms import test_val_transforms, train_transforms
from methods import ResNet18Model, ResNet50Model, EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, ResNext50
from settings import LOGS_DIR, CHECKPOINTS_DIR, PRETRAINED_NETS_DIR
from training import train_test


# experiment setup
SEED = 0
PROJECT_NAME = 'PretrainedResNet50'
NUM_CLASSES = 4
LR = 1e-4
BATCH_SIZE = 16
MAX_EPOCHS = 100
NORMALIZE = True
MONITOR = 'val_loss'
PATIENCE = 5
GPUS = -1
ENTITY_NAME = 'kn-bmi'
RESAMPLER = resamplers.identity_resampler
WEIGHTS = torch.Tensor([1, 1.5, 3, 1.5])
MODEL_PATH = '/home/adam_chlopowiec/data/eye_image_classification/Eye-disease/pretrained_nets/a9fefb3e60beb89d71f9f60f48189d43/TrainedResNet50Model.ckpt'
TEST_ONLY = True


models_list = [
        #EfficientNetB0(NUM_CLASSES),
        #EfficientNetB2(NUM_CLASSES),
        #ResNext50(2)
        #ResNet18Model(NUM_CLASSES),
        ResNet50Model(2)
    ]


def load_model(model, mode: str = 'train'):
    in_features = model.get_last_layer().in_features
    out_features = model.get_last_layer().out_features
    if mode == 'train':
        if out_features > 2:
            model.set_last_layer(torch.nn.Linear(in_features, 2))
        model.load_state_dict(torch.load(MODEL_PATH))
        model.set_last_layer(torch.nn.Linear(in_features, NUM_CLASSES))
    elif mode == 'test':
        if out_features < NUM_CLASSES:
            model.set_last_layer(torch.nn.Linear(in_features, NUM_CLASSES))
        model.load_state_dict(torch.load(MODEL_PATH))
    return model


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


def main():
    COUNTER = 0
    seed_all(SEED)
    for model in models_list:

        run_id = hashlib.md5(
            bytes(str(time.time()), encoding='utf-8')
        ).hexdigest()

        checkpoints_run_dir = CHECKPOINTS_DIR / run_id
        pretrain_run_dir = PRETRAINED_NETS_DIR / run_id
        COUNTER += 1
        print(run_id, 'ZROBIONO', time.time(), 'COUNTER ', COUNTER)
        os.mkdir(checkpoints_run_dir)
        os.mkdir(pretrain_run_dir)
        data_module = EyeDiseaseDataModule(
            csv_path='/media/data/adam_chlopowiec/eye_image_classification/pretrain_collected_data_splits.csv',
            train_split_name='train',
            val_split_name='val',
            test_split_name='test',
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
            pretraining=False
        )
        data_module.prepare_data()

        hparams = {
            'dataset': type(data_module).__name__,
            'model_type': type(model).__name__,
            'lr': LR,
            'batch_size': BATCH_SIZE,
            'optimizer': 'Adam',
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
                monitor=MONITOR,
                dirpath=pretrain_run_dir,
                save_top_k=1,
                filename='Trained' + type(model).__name__,
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
            load_classifier=MODEL_PATH,
            test_only=TEST_ONLY
        )
        logger.experiment.finish()


if __name__ == '__main__':
    main()
