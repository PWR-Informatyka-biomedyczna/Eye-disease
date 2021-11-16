from typing import NoReturn, Iterable

import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, f1, precision, recall, auc

from methods import BaseModel


class Classifier(pl.LightningModule):

    def __init__(self,
                 model: BaseModel,
                 num_classes: int,
                 lr: float,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam
                 ):
        super(Classifier, self).__init__()
        # optimizer config
        self.optimizer = optimizer
        self.lr = lr
        # model config
        self.model = model
        # metrics config
        self.metrics = {}
        for key in ['val', 'test']:
            self.metrics[key] = {
                'accuracy': lambda x, y: accuracy(x, y),
                'precision': lambda x, y: precision(x, y, num_classes=num_classes),
                'recall': lambda x, y: recall(x, y, num_classes=num_classes),
                'f1_micro': lambda x, y: f1(x, y, num_classes=num_classes),
                'f1_macro': lambda x, y: f1(x, y, num_classes=num_classes, average='macro')
            }
        # criterion config
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out

    def training_step(self, batch: Iterable, batch_idx: int) -> torch.Tensor:
        y_pred = self.model(batch)
        loss = self.criterion(y_pred, batch['target'])
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Iterable, batch_idx: int) -> torch.Tensor:
        y_pred = self.model(batch)
        loss = self.criterion(y_pred, batch['target'])
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self._calculate_score(y_pred, batch['target'], split='val', on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch: Iterable, batch_idx: int) -> None:
        y_pred = self.model(batch)
        self._calculate_score(y_pred, batch['target'], split='test', on_step=False, on_epoch=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        return optimizer

    def _calculate_score(self, y_pred: torch.Tensor, y_true: torch.Tensor, split: str, on_step: bool,
                         on_epoch: bool) -> NoReturn:
        score = {}
        output = torch.softmax(y_pred, dim=1).cuda()
        for metric_name, metric in self.metrics[split].items():
            score[f'{split}_{metric_name}'] = metric(output, y_true)
        self.log_dict(score, on_step=on_step, on_epoch=on_epoch)
