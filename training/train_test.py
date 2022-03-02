from typing import List

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.plugins.training_type import DDPPlugin
from pytorch_lightning.plugins.training_type.parallel import ParallelPlugin

from methods import BaseModel, Classifier


def train_test(
        model: BaseModel,
        datamodule: pl.LightningDataModule,
        max_epochs: int,
        num_classes: int,
        lr: float,
        gpus: int,
        callbacks: List[Callback],
        logger: LightningLoggerBase,
        optimizer: torch.optim.Optimizer = None,
        precision: int = 32,
        strategy = 'dp',
        #strategy: ParallelPlugin = DDPPlugin(find_unused_parameters=False),
        weights: torch.Tensor = None,
        lr_scheduler: torch.optim.lr_scheduler = None,
        test_only: bool = False,
        cross_val: bool = True
        ):
    """
    Base experiment function
    :param model:
    :param datamodule:
    :param max_epochs:
    :param num_classes:
    :param lr:
    :param gpus:
    :param callbacks:
    :param logger:
    :return:
    """
    if isinstance(model, Classifier):
        module = model
    else:
        module = Classifier(
            model=model,
            num_classes=num_classes,
            lr=lr,
            optimizer=optimizer,
            weights=weights,
            lr_scheduler=lr_scheduler
        )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=gpus,
        callbacks=callbacks,
        logger=logger,
        precision=precision,
        strategy=strategy
    )
    if test_only:
        trainer.test(
            model=module,
            test_dataloaders=datamodule.test_dataloader()
        )
        return module.test_score
    elif cross_val:
        trainer.fit(
            model=module,
            train_dataloaders=datamodule.kfold_train_dataloader(),
            val_dataloaders=datamodule.kfold_val_dataloader()
        )
        return module.val_score
    else:
        trainer.fit(
            model=module,
            train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader()
        )
        trainer.test(
            test_dataloaders=datamodule.test_dataloader()
        )
        return module.test_score
