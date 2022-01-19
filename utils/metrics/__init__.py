from typing import Tuple
from warnings import warn

import numpy as np
import torch
from torch.nn.functional import one_hot
from sklearn.metrics import roc_auc_score


def mask(y_pred: torch.Tensor, y_true: torch.Tensor, current_class: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create mask allowing for calculating metrics with respect to particular class
    """
    new_y_pred = torch.where(y_pred == current_class, 1, 0)
    new_y_true = torch.where(y_true == current_class, 1, 0)
    return new_y_pred, new_y_true


def true_positive(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Calculate true positive ratio
    """
    return torch.logical_and(y_pred == y_true, y_pred).sum()


def true_negative(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Calculate true negative ratio
    """
    return torch.logical_and(y_pred == y_true, torch.logical_not(y_pred)).sum()


def false_positive(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Calculate false positive ratio
    """
    return torch.logical_and(y_pred != y_true, y_pred).sum()


def false_negative(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Calclate false negative ratio
    """
    return torch.logical_and(y_pred != y_true, torch.logical_not(y_pred)).sum()


def sensitivity(probas: torch.Tensor, y_true: torch.Tensor, current_class: int) -> torch.Tensor:
    """
    Calculate sensitivity:

    sp = TP / (TP + FN)
    """
    y_pred = torch.argmax(probas, axis=1)
    new_pred, new_true = mask(y_pred, y_true, current_class)
    tp = true_positive(new_pred, new_true)
    fn = false_negative(new_pred, new_true)
    return tp / (tp + fn + 1e-10)

def specificity(probas: torch.Tensor, y_true: torch.Tensor, current_class: int) -> torch.Tensor:
    """
    Calculate specificity:

    SPF = TN / (TN + FP)
    """
    y_pred = torch.argmax(probas, axis=1)
    new_pred, new_true = mask(y_pred, y_true, current_class)
    tn = true_negative(new_pred, new_true)
    fp = false_positive(new_pred, new_true)
    return tn / (tn + fp + 1e-10)  

def f1_score(probas: torch.Tensor, y_true: torch.Tensor, current_class: int):
    """
    Calculate f1_score:

    F1 = 2 * TP / (2 * TP + FP + FN)
    """
    y_pred = torch.argmax(probas, axis=1)
    new_pred, new_true = mask(y_pred, y_true, current_class)
    tp = true_positive(new_pred, new_true)
    fp = false_positive(new_pred, new_true)
    fn = false_negative(new_pred, new_true)
    return 2 * tp / (2 * tp + fp + fn + 1e-10)   


def roc_auc(probas: torch.Tensor, y_true: torch.Tensor, strategy: str = 'ovr'):
    """
    Calculate ROC AUC
    """
    probas = probas.cpu()
    y_true = y_true.cpu()
    if len(y_true.unique()) > 1:
        y_true_one_hot = one_hot(y_true, probas.shape[1])
        print(y_true_one_hot)
        y_true_one_hot = np.asarray(y_true_one_hot[0])
        print(y_true_one_hot)
        return roc_auc_score(y_true_one_hot, probas, multi_class=strategy)
    return -1