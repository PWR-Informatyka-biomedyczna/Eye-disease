from experiments.common import seed_all
from typing import *
from collections import OrderedDict
from functools import partial

import os
import shutil
import math

import torch
from torch import distributed as dist
from torch.optim.lr_scheduler import LambdaLR



def lr_lambda(current_step, num_warmup_steps, num_training_steps, num_wait_steps, num_cycles):
    if current_step < num_wait_steps:
        return 0.0

    if current_step < num_warmup_steps + num_wait_steps:
        return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

    progress = float(current_step - num_warmup_steps - num_wait_steps) / \
        float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    
    lambd = partial(lr_lambda, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, num_wait_steps=num_wait_steps, num_cycles=num_cycles)
    return LambdaLR(optimizer, lambd, last_epoch)


def get_lr_scheduler(optimizer, hyperparams):
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=hyperparams["warmup_steps"],
        num_training_steps=hyperparams["total_steps"],
        num_wait_steps=0,
        num_cycles=1,
        last_epoch=-1
    )


def layer_wise_learning_rate_decay(model, hyperparams) -> List[Dict[str, torch.Tensor]]:
        names = []
        for _, (name, _) in enumerate(model.named_parameters()):
            names.append(name)
        lr = hyperparams["lr"]
        lr_decay = hyperparams["lr_decay"]
        parameters = []
        names.reverse()
        prev_group_name = names[0].split(".")[-2]
        for _, name in enumerate(names):
            cur_group_name = name.split(".")[-2]
            if cur_group_name != prev_group_name:
                lr *= lr_decay
            prev_group_name = cur_group_name
            no_decay = ["bn"]
            parameters += [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if n == name and p.requires_grad and (not any(nd in n for nd in no_decay))
                    ],
                    'weight_decay': hyperparams["weight_decay"],
                    "lr": lr,
                },
            ]
            parameters += [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if n == name and p.requires_grad and (any(nd in n for nd in no_decay))
                    ],
                    'weight_decay': 0.0,
                    "lr": lr,
                },
            ]
        return parameters
    

def save_checkpoint(state, is_best, hyperparams, finetune=False):
    """Imported from https://github.com/kekmodel/MPL-pytorch/blob/main/utils.py"""
    os.makedirs(hyperparams["save_path"], exist_ok=True)
    if finetune:
        name = f'{hyperparams["name"]}_finetune'
    else:
        name = hyperparams["name"]
    filename = f'{hyperparams["save_path"]}/{name}_last.pth.tar'
    torch.save(state, filename, _use_new_zipfile_serialization=True)
    if is_best:
        shutil.copyfile(filename, f'{hyperparams["save_path"]}/{hyperparams["name"]}_best.pth.tar')


def module_load_state_dict(model, state_dict):
    """Imported from https://github.com/kekmodel/MPL-pytorch/blob/main/utils.py"""
    try:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    except:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = f'module.{k}'  # add `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def model_load_state_dict(model, state_dict):
    """Imported from https://github.com/kekmodel/MPL-pytorch/blob/main/utils.py"""
    try:
        model.load_state_dict(state_dict)
    except:
        module_load_state_dict(model, state_dict)

    
def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

        
class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
       https://github.com/kekmodel/MPL-pytorch/blob/main/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def reduce_tensor(tensor, n):
    """Imported from https://github.com/kekmodel/MPL-pytorch/blob/main/utils.py"""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt
