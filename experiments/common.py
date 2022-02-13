import numpy as np
import torch
import random

from methods import Classifier


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        

def load_lightning_model(model_, lr_, num_classes_, ckpt_path):
    classifier = Classifier(
                model=model_,
                num_classes=num_classes_,
                lr=lr_,
                weights=torch.Tensor([1, 1]),
                optimizer=None,
                lr_scheduler=None
                )
    classifier.load_from_checkpoint(checkpoint_path=ckpt_path, model=model_, 
                                        num_classes=num_classes_, lr=lr_, weights=torch.Tensor([1, 1]), optimizer=None, lr_scheduler=None)
    return classifier.model


def freeze(model, n_params):
    i = 0
    for param in model.parameters():
        if i < n_params:
            param.requires_grad = False
            i += 1
    
    
def get_train_params_count(model):
    i = 0
    for _ in model.parameters():
        i += 1
    return i


def unfreeze(model, n_params):
    i = 0
    n_params = (int)(n_params)
    n = get_train_params_count(model)
    start = n - n_params
    for param in model.parameters():
        if start <= i and i <= n:
            param.requires_grad = True
        i += 1