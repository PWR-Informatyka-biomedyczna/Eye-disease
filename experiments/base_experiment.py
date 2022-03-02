import hashlib
import time
from pathlib import Path

import cv2
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


from experiments.common import seed_all, freeze, unfreeze, get_train_params_count, load_lightning_model
from utils.metrics import mean_metrics
from dataset import EyeDiseaseDataModule, resamplers
from dataset.transforms import test_val_transforms, train_transforms
from methods import RegNetY3_2gf
from settings import LOGS_DIR, CHECKPOINTS_DIR, PROJECT_DIR
from training import train_test
from dataset.resamplers import threshold_to_glaucoma_with_ros, binary_thresh_to_20k_equal, identity_resampler

SEED = 0
PROJECT_NAME = 'EyeDiseaseExperiments'
NUM_CLASSES = 2
LR = 3e-4
OPTIM = 'radam'
BATCH_SIZE = 24
MAX_EPOCHS = 200
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
MODEL_PATH = ""
k_folds = 5
n_runs = 1

models = [RegNetY3_2gf]

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
    for model_class in models:
        metrics = []
        for _ in range(n_runs):
            run_metrics = []
            model = model_class(NUM_CLASSES)
            if MODEL_PATH is not None:
                model = load_lightning_model(model, LR, NUM_CLASSES, MODEL_PATH)
            weights = torch.Tensor([1, 2])
            unfreeze(model, get_train_params_count(model))
            freeze(model, get_train_params_count(model) * (1/2))
            optimizer = init_optimizer(model, OPTIM, lr=LR)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
            
            run_id = hashlib.md5(
                    bytes(str(time.time()), encoding='utf-8')
                ).hexdigest()
            checkpoints_run_dir = CHECKPOINTS_DIR / run_id
            
            Path(checkpoints_run_dir).mkdir(mode=777, parents=True, exist_ok=True)
            
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
                binary=BINARY
            )
            data_module.prepare_data_kfold()

            hparams = {
                'dataset': type(data_module).__name__,
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
            for _ in range(k_folds):
                fold_metrics = train_test(
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
                run_metrics.append(fold_metrics)
            
            cross_val_test_score = train_test(
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
                    test_only=True
                )
            
            # TODO: Sprawdzić czy WandBLogger też ma metodę log_metrics, cz jakąś inną
            logger.log_metrics(cross_val_test_score)
            cross_val_metrics = mean_metrics(metrics=run_metrics, prefix="cross", sep="_")
            logger.log_metrics(cross_val_metrics)
            logger.experiment.finish()
            all_run_metrics = cross_val_metrics
            all_run_metrics.update(cross_val_test_score)
            metrics.append(all_run_metrics)
            
        avg_metrics = mean_metrics(metrics=metrics, prefix="avg", sep="_")
        logger = WandbLogger(
                save_dir=LOGS_DIR,
                config=hparams,
                project=PROJECT_NAME,
                log_model=False,
                entity=ENTITY_NAME
            )
        logger.log_metrics(avg_metrics)
        logger.log_hyperparams({"Path": run_id})
        logger.experiment.finish()


if __name__ == '__main__':
    main()
