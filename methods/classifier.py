from typing import NoReturn, Dict

import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, f1, auc
from utils.metrics import f1_score, sensitivity, specificity

from methods import BaseModel


class Classifier(pl.LightningModule):

    def __init__(self,
                 model: BaseModel,
                 num_classes: int,
                 lr: float,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam
                 ):
        """
        Base class for classifying task
        :param model: BaseModel
            neural network to train
        :param num_classes: int
            number of classes in task
        :param lr: float
            learning rate
        :param optimizer: torch.optim.Optimizer
            optimizer
        """
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
                'f1_micro': lambda x, y: f1(x, y, num_classes=num_classes),
                'f1_macro': lambda x, y: f1(x, y, num_classes=num_classes, average='macro'),
                'roc_auc': lambda x, y: auc(x, y)
            }
            for cls in range(num_classes):
                self.metrics[key] = {
                    f'f1_class_{cls}': lambda x, y: f1_score(x, y, current_class=cls),
                    f'sensitivity_class_{cls}': lambda x, y: sensitivity(x, y, current_class=cls),
                    f'specificity_class_{cls}': lambda x, y: sensitivity(x, y, current_class=cls)
                }
        # criterion config
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Return forward of model
        :param x: Dict[str, torch.Tensor]
            input to the model
        :return: torch.Tensor
            prediction of the model
        """
        out = self.model(x)
        return out

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step of model
        :param batch: Dict[str, torch.Tensor]
            batch of data
        :param batch_idx: int
            index of data
        :return: torch.Tensor
            loss in step
        """
        y_pred = self.model(batch)
        loss = self.criterion(y_pred, batch['target'])
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Validation step of model
        :param batch: Dict[str, torch.Tensor]
            batch of data
        :param batch_idx: int
            index of data
        :return: torch.Tensor
            loss in step
        """
        y_pred = self.model(batch)
        loss = self.criterion(y_pred, batch['target'])
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self._calculate_score(y_pred, batch['target'], split='val', on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Test step of model
        :param batch: Dict[str, torch.Tensor]
            batch of data
        :param batch_idx: int
            index of data
        :return:
        """
        y_pred = self.model(batch)
        self._calculate_score(y_pred, batch['target'], split='test', on_step=False, on_epoch=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizer and returns it
        :return: torch optimizer
        """
        optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        return optimizer

    def _calculate_score(self, y_pred: torch.Tensor, y_true: torch.Tensor, split: str, on_step: bool,
                         on_epoch: bool) -> NoReturn:
        """
        Calculate metrics for val and test set
        :param y_pred: torch.Tensor
            prediction of the model
        :param y_true: torch.Tensor
            real label
        :param split: str
            name of split
        :param on_step: bool
            are metrics calculated on step
        :param on_epoch:
            are metrics calculated on epoch
        :return:
        """
        score = {}
        output = torch.softmax(y_pred, dim=1).cuda()
        for metric_name, metric in self.metrics[split].items():
            score[f'{split}_{metric_name}'] = metric(output, y_true)
        self.log_dict(score, on_step=on_step, on_epoch=on_epoch)
