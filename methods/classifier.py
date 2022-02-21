from typing import NoReturn, Dict
from functools import partial

import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, f1
from utils.metrics import f1_score, sensitivity, specificity, roc_auc

from methods.base_model import BaseModel


class Classifier(pl.LightningModule):

    def __init__(self,
                 model: BaseModel,
                 num_classes: int,
                 lr: float,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 weights: torch.Tensor = None,
                 lr_scheduler: torch.optim.lr_scheduler = None
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
        self.lr_scheduler = lr_scheduler
        self.lr = lr
        # model config
        self.model = model
        # metrics config
        self.metrics = {}
        for key in ['val', 'test']:
            f1_micro = partial(f1, num_classes=num_classes)
            f1_macro = partial(f1, num_classes=num_classes, average='macro')
            #roc_auc_ovr = partial(roc_auc, strategy='ovr')
            self.metrics[key] = {
                'accuracy': accuracy,
                'f1_micro': f1_micro,
                'f1_macro': f1_macro,
            }
            f1_funcs = []
            sensitivity_funcs = []
            specificity_funcs = []
            roc_auc_funcs = []
            for cls in range(num_classes):
                f1_funcs.append(partial(f1_score, current_class=cls))
                sensitivity_funcs.append(partial(sensitivity, current_class=cls))
                specificity_funcs.append(partial(specificity, current_class=cls))
                roc_auc_funcs.append(partial(roc_auc, current_cls=cls))

            for f1_fun, sens, spec, roc_auc_f, cls in zip(f1_funcs, sensitivity_funcs, specificity_funcs, roc_auc_funcs, range(num_classes)):
                self.metrics[key][f'f1_class_{cls}'] = f1_fun
                self.metrics[key][f'sensitivity_class_{cls}'] = sens
                self.metrics[key][f'specificity_class_{cls}'] = spec
                self.metrics[key][f'roc_auc_class_{cls}'] = roc_auc_f
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        self.dicts_to_log = []
        self.test_dicts = []

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
        dict_to_log = {'pred': y_pred, 'target': batch['target']}
        self.dicts_to_log.append(dict_to_log)
        return loss

    def validation_epoch_end(self, validation_outputs):
        all_preds = []
        all_targets = []
        for output in self.dicts_to_log:
            for pred in output['pred']:
                all_preds.append(pred)
            for target in output['target']:
                all_targets.append(target)
        all_preds = torch.stack(all_preds)
        all_targets = torch.stack(all_targets)
        self._calculate_score(all_preds, all_targets, split='val', on_step=False, on_epoch=True)
        self.dicts_to_log = []
        
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
        dict_to_log = {'pred': y_pred, 'target': batch['target']}
        self.test_dicts.append(dict_to_log)
        
    def test_epoch_end(self, test_outputs):
        all_preds = []
        all_targets = []
        for output in self.test_dicts:
            for pred in output['pred']:
                all_preds.append(pred)
            for target in output['target']:
                all_targets.append(target)
        all_preds = torch.stack(all_preds)
        all_targets = torch.stack(all_targets)
        self._calculate_score(all_preds, all_targets, split='test', on_step=False, on_epoch=True)
        self.test_dicts = []

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizer and returns it
        :return: torch optimizer
        """
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        if self.lr_scheduler is not None:
            return [self.optimizer], [self.lr_scheduler]
        return self.optimizer

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
