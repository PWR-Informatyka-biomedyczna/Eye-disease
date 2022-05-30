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
        label_smoothing: float = 0.0,
        test: bool = True
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
            lr_scheduler=lr_scheduler,
            label_smoothing=label_smoothing
        )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=gpus,
        callbacks=callbacks,
        logger=logger
    )
    trainer.fit(
        model=module,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader()
    )
    if test:
        trainer.test(
            dataloaders=[datamodule.test_dataloader()]
        )
    best_model = Classifier.load_from_checkpoint(checkpoint_path=callbacks[0].best_model_path)
    logger.experiment.finish()
    return best_model.model
