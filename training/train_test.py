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
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        precision: int = 32,
        strategy: ParallelPlugin = DDPPlugin(find_unused_parameters=False)):
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
    module = Classifier(
        model=model,
        num_classes=num_classes,
        lr=lr,
        optimizer=optimizer
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=gpus,
        callbacks=callbacks,
        logger=logger,
        precison=precision,
        strategy=strategy
    )
    trainer.fit(
        model=module,
        train_dataloader=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader()
    )
    trainer.test(
        test_dataloaders=datamodule.test_dataloader()
    )
