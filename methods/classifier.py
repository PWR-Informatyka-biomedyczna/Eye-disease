from typing import NoReturn

import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1, AUC


class Classifier(pl.LightningModule):

    def __init__(self, model: torch.Module(), num_classes: int, lr: float):
        super(Classifier, self).__init__()
        self.model = model
        self.metrics = {}
        for key in ['val', 'test']:
            self.metrics[key] = {
                'accuracy': Accuracy(),
                'precision': Precision(num_classes=num_classes),
                'recall': Recall(num_classes=num_classes),
                'f1_micro': F1(num_classes=num_classes),
                'f1_macro': F1(num_classes=num_classes, average='macro'),
                'auc': AUC()
            }
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y_true = batch

        y_pred = self.model(x)
        loss = self.criterion(y_pred, y_true)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        x, y_true = batch

        y_pred = self.model(x)
        loss = self.criterion(y_pred, y_true)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self._calculate_score(y_pred, y_true, split='val', on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx) -> None:
        x, y_true = batch

        y_pred = self.model(x)
        self._calculate_score(y_pred, y_true, split='test', on_step=False, on_epoch=True)

    def configure_optimizers(self) -> torch.optim.Adam:
        optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        return optimizer

    def _calculate_score(self, y_pred: torch.Tensor, y_true: torch.Tensor, split: str, on_step: bool,
                         on_epoch: bool) -> NoReturn:
        score = {}
        output = torch.softmax(y_pred, dim=1)
        for metric_name, metric in self.metrics[split].items():
            score[f'{split}_{metric_name}'] = metric(output, y_true)
        self.log_dict(score, on_step=on_step, on_epoch=on_epoch)
