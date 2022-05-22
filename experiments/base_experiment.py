import hashlib
import time
from pathlib import Path

import cv2
import torch
import timm
from timm.scheduler.cosine_lr import CosineLRScheduler
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from adamp import AdamP


from experiments.common import seed_all, freeze, unfreeze, get_train_params_count, load_lightning_model, layer_decay
from utils.metrics import mean_metrics
from utils.callbacks import EMA
from dataset import EyeDiseaseDataModule, resamplers
from dataset.transforms import test_val_transforms, train_transforms
from methods import RegNetY3_2gf, ResNet50Model, ConvNextTiny
from settings import LOGS_DIR, CHECKPOINTS_DIR, PROJECT_DIR
from training import train_test
from dataset.resamplers import threshold_to_glaucoma_with_ros, binary_thresh_to_20k_equal, identity_resampler

SEED = 0
PROJECT_NAME = 'EyeDiseaseExperimentsRemastered'
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

models = [RegNetY3_2gf, ConvNextTiny, ResNet50Model]

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


def main():
    seed_all(SEED)
    # Wagi zaleznie czy pretraining czy finetuning
    #weights = torch.Tensor([0.408, 0.408, 1, 0.408])
    weights = torch.Tensor([0.647, 1])
    for model_class in models:
        run_id = hashlib.md5(
            bytes(str(time.time()), encoding='utf-8')
        ).hexdigest()
        metrics = []
        for i in range(n_runs):
            run_metrics = []
            model = model_class(NUM_CLASSES)
            if MODEL_PATH is not None:
                model = load_lightning_model(model, LR, NUM_CLASSES, MODEL_PATH)
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
            #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
            
            experiment_id = run_id + '_' + str(i)
            checkpoints_run_dir = CHECKPOINTS_DIR / experiment_id
            
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
            if cross_val:
                data_module.prepare_data_kfold()
            else:
                data_module.prepare_data()

            hparams = {
                'dataset': type(data_module).__name__,
                'model_type': type(model).__name__,
                'lr': LR,
                'batch_size': BATCH_SIZE,
                'optimizer': type(optimizer).__name__,
                'resampler': RESAMPLER.__name__,
                'num_classes': NUM_CLASSES,
                'run_id': experiment_id
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
            if EMA_DECAY > 0:
                callbacks.append(
                    EMA(EMA_DECAY)
                )
            if cross_val:
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
                        test_only=TEST_ONLY,
                        cross_val=cross_val,
                        label_smoothing=LABEL_SMOOTHING
                    )
                    run_metrics.append(fold_metrics)
            else:
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
                    test_only=TEST_ONLY,
                    cross_val=cross_val,
                    label_smoothing=LABEL_SMOOTHING
                )
                run_metrics.append(fold_metrics)

            if cross_val_test:
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
                        test_only=True,
                        label_smoothing=LABEL_SMOOTHING
                    )
                logger.log_metrics(cross_val_test_score)
            if cross_val:
                cross_val_metrics = mean_metrics(metrics=run_metrics, prefix="cross", sep="_")
                logger.log_metrics(cross_val_metrics)
                logger.experiment.finish()
                all_run_metrics = cross_val_metrics
                if cross_val_test:
                    all_run_metrics.update(cross_val_test_score)
            else:
                all_run_metrics = run_metrics[0]
                logger.log_metrics(all_run_metrics)
                logger.experiment.finish()
            metrics.append(all_run_metrics)

        if n_runs > 1:
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
