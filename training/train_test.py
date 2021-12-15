from typing import List

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.plugins.training_type import DDPPlugin
from pytorch_lightning.plugins.training_type.parallel import ParallelPlugin

from methods import BaseModel, Classifier
from methods import EfficientNetB2


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
        strategy: ParallelPlugin = DDPPlugin(find_unused_parameters=False),
        weights: torch.Tensor = None,
        load_classifier: str = None,
        test_only: bool = False
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
    if load_classifier is not None:
        model = model.__class__(4)
        module = Classifier.load_from_checkpoint(load_classifier, model=model, num_classes=4, lr=1e-4, weights=weights[:4])
        #module.model.set_last_layer(torch.nn.Linear(model.get_last_layer().in_features, 4))
        module.criterion.weight = weights
    else:
        module = Classifier(
            model=model,
            num_classes=num_classes,
            lr=lr,
            optimizer=optimizer,
            weights=weights
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
    else:
        trainer.fit(
            model=module,
            train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader()
        )
        trainer.test(
            test_dataloaders=datamodule.test_dataloader()
        )
