from typing import Tuple
import torch


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


def sensitivity(logits: torch.Tensor, y_true: torch.Tensor, current_class: int) -> torch.Tensor:
    """
    Calculate sensitivity:

    sp = TP / (TP + FN)
    """
    y_pred = torch.argmax(logits, axis=1)
    new_pred, new_true = mask(y_pred, y_true, current_class)
    tp = true_positive(new_pred, new_true)
    fn = false_negative(new_pred, new_true)
    return tp / (tp + fn + 1e-10)

def specificity(logits: torch.Tensor, y_true: torch.Tensor, current_class: int) -> torch.Tensor:
    """
    Calculate specificity:

    SPF = TN / (TN + FP)
    """
    y_pred = torch.argmax(logits, axis=1)
    new_pred, new_true = mask(y_pred, y_true, current_class)
    tn = true_negative(new_pred, new_true)
    fp = false_positive(new_pred, new_true)
    return tn / (tn + fp + 1e-10)  

def f1_score(logits: torch.Tensor, y_true: torch.Tensor, current_class: int):
    """
    Calculate f1_score:

    F1 = 2 * TP / (2 * TP + FP + FN)
    """
    y_pred = torch.argmax(logits, axis=1)
    new_pred, new_true = mask(y_pred, y_true, current_class)
    tp = true_positive(new_pred, new_true)
    fp = false_positive(new_pred, new_true)
    fn = false_negative(new_pred, new_true)
    return 2 * tp / (2 * tp + fp + fn + 1e-10)   
