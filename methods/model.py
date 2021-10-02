from typing import NoReturn

import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1, AUC


class Model(pl.LightningModule):

    def __init__(self, model: torch.Module(), num_classes: int, lr: float):
        super(Model, self).__init__()
        self._model = model
        self._metrics = {
            'accuracy': Accuracy(),
            'precision': Precision(num_classes=num_classes, average=None),
            'recall': Recall(num_classes=num_classes, average=None),
            'f1_micro': F1(num_classes=num_classes, average=None),
            'f1_macro': F1(num_classes=num_classes, average='macro'),
            'auc': AUC()
        }
        self._lr = lr
        self._criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._model(x)
        return out

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y_true = batch

        y_pred = self._model(x)
        loss = self._criterion(y_pred, y_true)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        x, y_true = batch

        y_pred = self._model(x)
        loss = self._criterion(y_pred, y_true)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self._calculate_score(y_pred, y_true, split='val', on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx) -> None:
        x, y_true = batch

        y_pred = self._model(x)
        self._calculate_score(y_pred, y_true, split='test', on_step=False, on_epoch=True)

    def configure_optimizers(self) -> torch.optim.Adam:
        optimizer = torch.optim.Adam(self._model.parameters(), self._lr)
        return optimizer

    def _calculate_score(self, y_pred: torch.Tensor, y_true: torch.Tensor, split: str, on_step: bool,
                         on_epoch: bool) -> NoReturn:
        score = {}
        output = torch.softmax(y_pred, dim=1)
        for metric_name, metric in self._metrics.items():
            score[f'{split}_{metric_name}'] = metric(output, y_true)
        self.log_dict(score, on_step=on_step, on_epoch=on_epoch)
