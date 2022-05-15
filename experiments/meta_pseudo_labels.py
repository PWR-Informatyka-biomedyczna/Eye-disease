from experiments.common import seed_all
from typing import *
import timm
from timm.scheduler.cosine_lr import CosineLRScheduler
from collections import OrderedDict
from functools import partial

import logging
import time
import os
import shutil
import math

import torch
from torch import distributed as dist
from torch.cuda import amp
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
import wandb
from tqdm import tqdm
from utils.metrics import f1_score, sensitivity, specificity, roc_auc
from torchmetrics.functional import accuracy
from torchmetrics.functional import f1_score as f1
from adamp import AdamP
from dataset import EyeDiseaseDataModule
from dataset.transforms import train_transforms, test_val_transforms, mpl_transforms
from dataset.resamplers import identity_resampler
from methods import RegNetY3_2gf, RegNetY400MF
import cv2

logger = logging.getLogger(__name__)

hyperparams = {
    "project_name": "AIProjectMetaPseudoLabels",
    "entity_name": "kn-bmi",
    "seed": 42,
    "t_initial": 30,
    "eta_min": 1e-7,
    "warmup_lr_init": 1e-5,
    "warmup_steps": 5000,
    "cycle_mul": 1,
    "cycle_decay": 0.1,
    "lr": 1e-4,
    "batch_size": 8,
    "lr_decay": 0.9,
    "weight_decay": 1e-2,
    "betas": (0.9, 0.999),
    "nesterov": True,
    "weights": torch.Tensor([0.5, 1, 1, 1]),
    "workers": 1,
    "world_size": 1,
    "start_step": 0,
    "total_steps": 50000,
    "eval_step": 300,
    "local_rank": -1,
    "gpu": 0,
    "device": torch.device('cuda', 0),
    "amp": True,
    "uda_temperature": 0.8,
    "threshold": 0.9,
    "uda_factor": 2.5,
    "uda_steps": 10000,
    "grad_clip": 0,
    "ema": False,
    "best_accuracy": 0.,
    "best_f1_macro": 0.,
    "save_path": r"C:\Users\Adam\Desktop\Studia\Psy Tabakowa\eye-disease\Eye-disease\checkpoints",
    "name": "regnet_y_3.2gf",
    "finetune_epochs": 5,
    "train_split_name": "train",
    "val_split_name": "val",
    "test_split_name": "test",
    "unlabeled_split_name": "pretrain",
    "normalize": True,
    "resampler": identity_resampler,
    "resume": False,
    "evaluate": False,
    "finetune": False
}


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


def layer_wise_learning_rate_decay(model) -> List[Dict[str, torch.Tensor]]:
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
    

def save_checkpoint(state, is_best, finetune=False):
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

def get_lr_scheduler(optimizer):
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=hyperparams["warmup_steps"],
        num_training_steps=hyperparams["total_steps"],
        num_wait_steps=0,
        num_cycles=1,
        last_epoch=-1
    )
    

def train_loop(labeled_loader, unlabeled_loader, val_loader, finetune_loader, teacher_model, 
               student_model, avg_student_model, criterion, t_optimizer, s_optimizer, t_scheduler, s_scheduler, t_scaler, s_scaler):
    """Mostly imported from https://github.com/kekmodel/MPL-pytorch/blob/main/main.py"""
    print("Train loop start")
    if hyperparams["world_size"] > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_loader.sampler.set_epoch(labeled_epoch)
        unlabeled_loader.sampler.set_epoch(unlabeled_epoch)
    
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
        
    pbar = tqdm(range(hyperparams["eval_step"]), disable=hyperparams["local_rank"] not in [-1, 0])
    batch_time = AverageMeter()
    data_time = AverageMeter()
    s_losses = AverageMeter()
    t_losses = AverageMeter()
    t_losses_l = AverageMeter()
    t_losses_u = AverageMeter()
    t_losses_mpl = AverageMeter()
    mean_mask = AverageMeter()
    
    for step in range(hyperparams["start_step"], hyperparams["total_steps"]):
        print("Step")
        if step % hyperparams["eval_step"] == 0:
            pbar = tqdm(range(hyperparams["eval_step"]), disable=hyperparams["local_rank"] not in [-1, 0])
            batch_time = AverageMeter()
            data_time = AverageMeter()
            s_losses = AverageMeter()
            t_losses = AverageMeter()
            t_losses_l = AverageMeter()
            t_losses_u = AverageMeter()
            t_losses_mpl = AverageMeter()
            mean_mask = AverageMeter()

        teacher_model.train()
        student_model.train()
        end = time.time()

        try:
            images_l, targets = labeled_iter.next()
        except:
            if hyperparams["world_size"] > 1:
                labeled_epoch += 1
                labeled_loader.sampler.set_epoch(labeled_epoch)
            labeled_iter = iter(labeled_loader)
            images_l, targets = labeled_iter.next()

        try:
            (images_uw, images_us), _ = unlabeled_iter.next()
        except:
            if hyperparams["world_size"] > 1:
                unlabeled_epoch += 1
                unlabeled_loader.sampler.set_epoch(unlabeled_epoch)
            unlabeled_iter = iter(unlabeled_loader)
            (images_uw, images_us), _ = unlabeled_iter.next()
        
        data_time.update(time.time() - end)
        print("Past img loading")

        images_l = images_l.to(hyperparams["device"])
        images_uw = images_uw.to(hyperparams["device"])
        images_us = images_us.to(hyperparams["device"])
        targets = targets.to(hyperparams["device"])
        with amp.autocast(enabled=hyperparams["amp"]):
            batch_size = images_l.shape[0]
            t_images = torch.cat((images_l, images_uw, images_us))
            t_logits = teacher_model(t_images)
            t_logits_l = t_logits[:batch_size]
            t_logits_uw, t_logits_us = t_logits[batch_size:].chunk(2)
            del t_logits

            t_loss_l = criterion(t_logits_l, targets)

            soft_pseudo_label = torch.softmax(t_logits_uw.detach() / hyperparams["uda_temperature"], dim=-1)
            max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
            mask = max_probs.ge(hyperparams["threshold"]).float()
            t_loss_u = torch.mean(
                -(soft_pseudo_label * torch.log_softmax(t_logits_us, dim=-1)).sum(dim=-1) * mask
            )
            weight_u = hyperparams["uda_factor"] * min(1., (step + 1) / hyperparams["uda_steps"])
            t_loss_uda = t_loss_l + weight_u * t_loss_u

            s_images = torch.cat((images_l, images_us))
            s_logits = student_model(s_images)
            s_logits_l = s_logits[:batch_size]
            s_logits_us = s_logits[batch_size:]
            del s_logits

            s_loss_l_old = F.cross_entropy(s_logits_l.detach(), targets)
            s_loss = criterion(s_logits_us, hard_pseudo_label)
        
        s_scaler.scale(s_loss).backward()
        if hyperparams["grad_clip"] > 0:
            s_scaler.unscale_(s_optimizer)
            nn.utils.clip_grad_norm_(student_model.parameters(), hyperparams["grad_clip"])
        s_scaler.step(s_optimizer)
        s_scaler.update()
        s_scheduler.step()
        if hyperparams["ema"] > 0:
            avg_student_model.update_parameters(student_model)
        print("Past student update")

        with amp.autocast(enabled=hyperparams["amp"]):
            with torch.no_grad():
                s_logits_l = student_model(images_l)
            s_loss_l_new = F.cross_entropy(s_logits_l.detach(), targets)

            # theoretically correct formula (https://github.com/kekmodel/MPL-pytorch/issues/6)
            # dot_product = s_loss_l_old - s_loss_l_new

            # author's code formula
            dot_product = s_loss_l_new - s_loss_l_old # Tutaj powinien byc cosine distance? Przydalaby sie pomoc Artura w analizie tego
            # moving_dot_product = moving_dot_product * 0.99 + dot_product * 0.01
            # dot_product = dot_product - moving_dot_product

            _, hard_pseudo_label = torch.max(t_logits_us.detach(), dim=-1)
            t_loss_mpl = dot_product * F.cross_entropy(t_logits_us, hard_pseudo_label)
            # test
            # t_loss_mpl = torch.tensor(0.).to(args.device)
            t_loss = t_loss_uda + t_loss_mpl

        t_scaler.scale(t_loss).backward()
        if hyperparams["grad_clip"] > 0:
            t_scaler.unscale_(t_optimizer)
            nn.utils.clip_grad_norm_(teacher_model.parameters(), hyperparams["grad_clip"])
        t_scaler.step(t_optimizer)
        t_scaler.update()
        t_scheduler.step()
        print("Past teacher update")

        teacher_model.zero_grad()
        student_model.zero_grad()

        if hyperparams["world_size"] > 1:
            s_loss = reduce_tensor(s_loss.detach(), hyperparams["world_size"])
            t_loss = reduce_tensor(t_loss.detach(), hyperparams["world_size"])
            t_loss_l = reduce_tensor(t_loss_l.detach(), hyperparams["world_size"])
            t_loss_u = reduce_tensor(t_loss_u.detach(), hyperparams["world_size"])
            t_loss_mpl = reduce_tensor(t_loss_mpl.detach(), hyperparams["world_size"])
            mask = reduce_tensor(mask, hyperparams["world_size"])

        s_losses.update(s_loss.item())
        t_losses.update(t_loss.item())
        t_losses_l.update(t_loss_l.item())
        t_losses_u.update(t_loss_u.item())
        t_losses_mpl.update(t_loss_mpl.item())
        mean_mask.update(mask.mean().item())

        batch_time.update(time.time() - end)
        pbar.set_description(
            f"Train Iter: {step+1:3}/{hyperparams['world_size']:3}. "
            f"LR: {hyperparams['lr']:.4f}. Data: {data_time.avg:.2f}s. "
            f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. "
            f"T_Loss: {t_losses.avg:.4f}. Mask: {mean_mask.avg:.4f}. ")
        pbar.update()
        if hyperparams["local_rank"] in [-1, 0]:
            wandb.log({"lr": hyperparams["lr"]})

        hyperparams["num_eval"] = step // hyperparams["eval_step"]
        if (step + 1) % hyperparams["eval_step"] == 0:
            pbar.close()
            if hyperparams["local_rank"] in [-1, 0]:
                wandb.log({"train_student_loss": s_losses.avg,
                           "train_teacher_loss": t_losses.avg,
                           "train_teacher_labeled_loss": t_losses_l.avg,
                           "train_teacher_unlabeled_loss": t_losses_u.avg,
                           "train_teacher_mpl_loss": t_losses_mpl.avg,
                           "train_uda_mask_mean": mean_mask.avg})

                val_model = avg_student_model if avg_student_model is not None else student_model
                val_loss, acc, f1_macro = evaluate(val_loader, val_model, criterion)

                wandb.log({"val_loss": val_loss,
                           "test_accuracy": acc,
                           "test_f1_macro": f1_macro})

                is_best = acc > hyperparams["best_f1_macro"]
                if is_best:
                    hyperparams["best_f1_macro"] = f1_macro
                    hyperparams["best_accuracy"] = acc

                logger.info(f"f1_macro: {f1_macro:.2f}")
                logger.info(f"Best f1_macro: {hyperparams['best_f1_macro']:.2f}")

                save_checkpoint({
                    'step': step + 1,
                    'teacher_state_dict': teacher_model.state_dict(),
                    'student_state_dict': student_model.state_dict(),
                    'avg_state_dict': avg_student_model.state_dict() if avg_student_model is not None else None,
                    'accuracy': hyperparams["accuracy"],
                    'f1_macro': hyperparams["f1_macro"],
                    'teacher_optimizer': t_optimizer.state_dict(),
                    'student_optimizer': s_optimizer.state_dict(),
                    'teacher_scheduler': t_scheduler.state_dict(),
                    'student_scheduler': s_scheduler.state_dict(),
                    'teacher_scaler': t_scaler.state_dict(),
                    'student_scaler': s_scaler.state_dict(),
                }, is_best)

    if hyperparams["local_rank"] in [-1, 0]:
        wandb.log({"best_val_acc": hyperparams["accuracy"]})
        wandb.log({"best_f1_macro": hyperparams["f1_macro"]})

    # finetune
    del t_scaler, t_scheduler, t_optimizer, teacher_model, labeled_loader, unlabeled_loader
    del s_scaler, s_scheduler, s_optimizer
    ckpt_name = f'{hyperparams["save_path"]}/{hyperparams["name"]}_best.pth.tar'
    loc = f'cuda:{hyperparams["gpu"]}'
    checkpoint = torch.load(ckpt_name, map_location=loc)
    logger.info(f"=> loading checkpoint '{ckpt_name}'")
    if checkpoint['avg_state_dict'] is not None:
        model_load_state_dict(student_model, checkpoint['avg_state_dict'])
    else:
        model_load_state_dict(student_model, checkpoint['student_state_dict'])
    finetune(finetune_loader, val_loader, student_model, criterion)
    return


def evaluate(val_loader, model, criterion):
    """Mostly imported from https://github.com/kekmodel/MPL-pytorch/blob/main/main.py"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc_meter = AverageMeter()
    f1_macro_meter = AverageMeter()
    model.eval()
    val_iter = tqdm(val_loader, disable=hyperparams["local_rank"] not in [-1, 0])
    with torch.no_grad():
        end = time.time()
        for step, (images, targets) in enumerate(val_iter):
            data_time.update(time.time() - end)
            batch_size = images.shape[0]
            images = images.to(hyperparams["device"])
            targets = targets.to(hyperparams["device"])
            with amp.autocast(enabled=hyperparams["amp"]):
                outputs = model(images)
                loss = criterion(outputs, targets)

            acc_res = accuracy(outputs, targets)
            f1_macro = f1(outputs, targets, average="macro")
            # acc = accuracy(outputs, targets)
            # sens_res = accuracy(outputs, targets)
            # spec_res = accuracy(outputs, targets)
            # roc_res = accuracy(outputs, targets)
            losses.update(loss.item(), batch_size)
            acc_meter.update(acc_res, batch_size)
            f1_macro_meter.update(f1_macro, batch_size)
            batch_time.update(time.time() - end)
            end = time.time()
            val_iter.set_description(
                f"Test Iter: {step+1:3}/{len(val_loader):3}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. "
                f"accuracy: {acc_meter.avg:.2f}. f1_macro: {f1_macro_meter.avg:.2f}. ")

        val_iter.close()
        return losses.avg, acc_meter.avg, f1_macro_meter.avg


def finetune(labeled_loader, val_loader, model, criterion):
    """Mostly imported from https://github.com/kekmodel/MPL-pytorch/blob/main/main.py"""
    # model.drop = nn.Identity()
    parameters = layer_wise_learning_rate_decay(model)
    optimizer = AdamP(
        parameters,
        lr=hyperparams["lr"],
        betas=(hyperparams["betas"][0], hyperparams["betas"][1]),
        weight_decay=hyperparams["weight_decay"],
        nesterov=hyperparams["nesterov"],
    )
    scheduler = get_lr_scheduler(optimizer)
    scaler = amp.GradScaler(enabled=hyperparams["amp"])

    logger.info("***** Running Finetuning *****")
    logger.info(f"   Finetuning steps = {len(labeled_loader)* hyperparams['finetune_epochs']}")

    for epoch in range(hyperparams['finetune_epochs']):
        if hyperparams['world_size'] > 1:
            labeled_loader.sampler.set_epoch(epoch + 624)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        model.train()
        end = time.time()
        labeled_iter = tqdm(labeled_loader, disable=hyperparams['local_rank'] not in [-1, 0])
        for step, (images, targets) in enumerate(labeled_iter):
            data_time.update(time.time() - end)
            batch_size = images.shape[0]
            images = images.to(hyperparams['device'])
            targets = targets.to(hyperparams['device'])
            with amp.autocast(enabled=hyperparams['amp']):
                model.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if hyperparams['world_size'] > 1:
                loss = reduce_tensor(loss.detach(), hyperparams['world_size'])
            losses.update(loss.item(), batch_size)
            batch_time.update(time.time() - end)
            labeled_iter.set_description(
                f"Finetune Epoch: {epoch+1:2}/{hyperparams['finetune_epochs']:2}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. ")
        labeled_iter.close()
        if hyperparams['local_rank'] in [-1, 0]:
            val_loss, acc, f1_macro = evaluate(val_loader, model, criterion)
            wandb.log({"finetune_train_loss": losses.avg,
                       "finetune_val_loss": val_loss,
                       "finetune_accuracy": acc,
                       "finetune_f1_macro": f1_macro})

            is_best = f1_macro > hyperparams["f1_macro"]
            if is_best:
                hyperparams["f1_macro"] = f1_macro
                hyperparams["accuracy"] = acc

            logger.info(f"f1_macro: {f1_macro:.2f}")
            logger.info(f"Best f1_macro: {hyperparams['f1_macro']:.2f}")

            save_checkpoint({
                'step': step + 1,
                'best_accuracy': hyperparams['accuracy'],
                'best_f1_macro': hyperparams['f1_macro'],
                'student_state_dict': model.state_dict(),
                'avg_state_dict': None,
                'student_optimizer': optimizer.state_dict(),
            }, is_best, finetune=True)
        if hyperparams['local_rank'] in [-1, 0]:
            wandb.log({"best_finetune_f1_macro": hyperparams['f1_macro']})
            wandb.log({"best_finetune_accuracy": hyperparams['accuracy']})
    return


def main():
    """Mostly imported from https://github.com/kekmodel/MPL-pytorch/blob/main/main.py"""
    if hyperparams["local_rank"] != -1:
        hyperparams["gpu"] = hyperparams["local_rank"]
        torch.distributed.init_process_group(backend='nccl')
        hyperparams["world_size"] = torch.distributed.get_world_size()
    else:
        hyperparams["gpu"] = 0
        hyperparams["world_size"] = 1

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if hyperparams["local_rank"] in [-1, 0] else logging.WARNING)

    logger.warning(
        f"Process rank: {hyperparams['local_rank']}, "
        f"device: {hyperparams['device']}, "
        f"distributed training: {bool(hyperparams['local_rank'] != -1)}, "
        f"16-bits training: {hyperparams['amp']}")

    logger.info(hyperparams)

    if hyperparams["local_rank"] in [-1, 0]:
        wandb.init(name=hyperparams["name"], project=hyperparams["project_name"], entity=hyperparams["entity_name"], config=hyperparams)

    if hyperparams["local_rank"] not in [-1, 0]:
        torch.distributed.barrier()

    teacher_model = RegNetY400MF(4)
    student_model = RegNetY400MF(4)
    labeled_datamodule = EyeDiseaseDataModule(
                csv_path=r'C:\Users\Adam\Desktop\Studia\Psy Tabakowa\eye-disease\data\pretrain_corrected_data_splits_windows.csv',
                train_split_name=hyperparams["train_split_name"],
                val_split_name=hyperparams["val_split_name"],
                test_split_name=hyperparams["test_split_name"],
                train_transforms=test_val_transforms(teacher_model.input_size, hyperparams["normalize"], cv2.INTER_NEAREST),
                val_transforms=test_val_transforms(teacher_model.input_size, hyperparams["normalize"], cv2.INTER_NEAREST),
                test_transforms=test_val_transforms(teacher_model.input_size, hyperparams["normalize"], cv2.INTER_NEAREST),
                image_path_name='Path',
                target_name='Label',
                split_name='Split',
                batch_size=hyperparams["batch_size"],
                num_workers=12,
                shuffle_train=True,
                resampler=identity_resampler,
                pretraining=False,
                binary=False
    )
    labeled_datamodule.prepare_data()
    
    unlabeled_datamodule = EyeDiseaseDataModule(
                csv_path=r'C:\Users\Adam\Desktop\Studia\Psy Tabakowa\eye-disease\data\pretrain_corrected_data_splits_windows.csv',
                train_split_name=hyperparams["unlabeled_split_name"],
                val_split_name=hyperparams["val_split_name"],
                test_split_name=hyperparams["test_split_name"],
                train_transforms=mpl_transforms(teacher_model.input_size, hyperparams["normalize"], cv2.INTER_NEAREST),
                val_transforms=test_val_transforms(teacher_model.input_size, hyperparams["normalize"], cv2.INTER_NEAREST),
                test_transforms=test_val_transforms(teacher_model.input_size, hyperparams["normalize"], cv2.INTER_NEAREST),
                image_path_name='Path',
                target_name='Label',
                split_name='Split',
                batch_size=hyperparams["batch_size"],
                num_workers=12,
                shuffle_train=True,
                resampler=identity_resampler,
                pretraining=False,
                binary=False
    )
    unlabeled_datamodule.prepare_data()
    
    finetune_datamodule = EyeDiseaseDataModule(
                csv_path=r'C:\Users\Adam\Desktop\Studia\Psy Tabakowa\eye-disease\data\pretrain_corrected_data_splits_windows.csv',
                train_split_name=hyperparams["train_split_name"],
                val_split_name=hyperparams["val_split_name"],
                test_split_name=hyperparams["test_split_name"],
                train_transforms=train_transforms(teacher_model.input_size, hyperparams["normalize"], cv2.INTER_NEAREST),
                val_transforms=test_val_transforms(teacher_model.input_size, hyperparams["normalize"], cv2.INTER_NEAREST),
                test_transforms=test_val_transforms(teacher_model.input_size, hyperparams["normalize"], cv2.INTER_NEAREST),
                image_path_name='Path',
                target_name='Label',
                split_name='Split',
                batch_size=hyperparams["batch_size"],
                num_workers=12,
                shuffle_train=True,
                resampler=identity_resampler,
                pretraining=False,
                binary=False
    )
    finetune_datamodule.prepare_data()

    if hyperparams["local_rank"] == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if hyperparams["local_rank"] == -1 else DistributedSampler
    labeled_loader = labeled_datamodule.train_dataloader()

    unlabeled_loader = unlabeled_datamodule.train_dataloader()
    val_loader = labeled_datamodule.val_dataloader()

    if hyperparams["local_rank"] not in [-1, 0]:
        torch.distributed.barrier()

    if hyperparams["local_rank"] == 0:
        torch.distributed.barrier()

    logger.info(f"Teacher Model: {teacher_model.__class__}")
    logger.info(f"Student Model: {student_model.__class__}")
    logger.info(f"Params: {sum(p.numel() for p in teacher_model.parameters())/1e6:.2f}M")

    teacher_model.to(hyperparams["device"])
    student_model.to(hyperparams["device"])
    avg_student_model = None
    # if hyperparams["ema"] > 0:
    #     avg_student_model = ModelEMA(student_model, args.ema)

    criterion = torch.nn.CrossEntropyLoss(weight=hyperparams["weights"])

    no_decay = ['bn']
    teacher_parameters = layer_wise_learning_rate_decay(teacher_model)
    student_parameters = layer_wise_learning_rate_decay(student_model)

    teacher_optimizer = AdamP(
        teacher_parameters,
        lr=hyperparams["lr"],
        betas=(hyperparams["betas"][0], hyperparams["betas"][1]),
        weight_decay=hyperparams["weight_decay"],
        nesterov=hyperparams["nesterov"],
    )
    teacher_scheduler = get_lr_scheduler(teacher_optimizer)
    
    student_optimizer = AdamP(
        student_parameters,
        lr=hyperparams["lr"],
        betas=(hyperparams["betas"][0], hyperparams["betas"][1]),
        weight_decay=hyperparams["weight_decay"],
        nesterov=hyperparams["nesterov"],
    )
    student_scheduler = get_lr_scheduler(student_optimizer)

    teacher_scaler = amp.GradScaler(enabled=hyperparams["amp"])
    student_scaler = amp.GradScaler(enabled=hyperparams["amp"])

    # optionally resume from a checkpoint
    if hyperparams["resume"]:
        if os.path.isfile(hyperparams['resume']):
            logger.info(f"=> loading checkpoint '{hyperparams['resume']}'")
            loc = f"cuda:{hyperparams['gpu']}"
            checkpoint = torch.load(hyperparams['resume'], map_location=loc)
            hyperparams['best_accuracy'] = checkpoint['best_accuracy'].to(torch.device('cpu'))
            hyperparams['best_f1_macro'] = checkpoint['best_f1_macro'].to(torch.device('cpu'))
            if not (hyperparams["evaluate"] or hyperparams["finetune"]):
                hyperparams['start_step'] = checkpoint['step']
                teacher_optimizer.load_state_dict(checkpoint['teacher_optimizer'])
                student_optimizer.load_state_dict(checkpoint['student_optimizer'])
                teacher_scheduler.load_state_dict(checkpoint['teacher_scheduler'])
                student_scheduler.load_state_dict(checkpoint['student_scheduler'])
                teacher_scaler.load_state_dict(checkpoint['teacher_scaler'])
                student_scaler.load_state_dict(checkpoint['student_scaler'])
                model_load_state_dict(teacher_model, checkpoint['teacher_state_dict'])
                if avg_student_model is not None:
                    model_load_state_dict(avg_student_model, checkpoint['avg_state_dict'])

            else:
                if checkpoint['avg_state_dict'] is not None:
                    model_load_state_dict(student_model, checkpoint['avg_state_dict'])
                else:
                    model_load_state_dict(student_model, checkpoint['student_state_dict'])

            logger.info(f"=> loaded checkpoint '{hyperparams['resume']}' (step {checkpoint['step']})")
        else:
            logger.info(f"=> no checkpoint found at '{hyperparams['resume']}'")

    if hyperparams['local_rank'] != -1:
        teacher_model = nn.parallel.DistributedDataParallel(
            teacher_model, device_ids=[hyperparams['local_rank']],
            output_device=hyperparams['local_rank'], find_unused_parameters=True)
        student_model = nn.parallel.DistributedDataParallel(
            student_model, device_ids=[hyperparams['local_rank']],
            output_device=hyperparams['local_rank'], find_unused_parameters=True)

    if hyperparams['finetune']:
        del t_scaler, t_scheduler, t_optimizer, teacher_model, unlabeled_loader
        del s_scaler, s_scheduler, s_optimizer
        finetune(finetune_datamodule.train_dataloader(), val_loader, student_model, criterion)
        return

    if hyperparams['evaluate']:
        del t_scaler, t_scheduler, t_optimizer, teacher_model, unlabeled_loader, labeled_loader
        del s_scaler, s_scheduler, s_optimizer
        evaluate(val_loader, student_model, criterion)
        return

    teacher_model.zero_grad()
    student_model.zero_grad()
    train_loop(labeled_loader, unlabeled_loader, val_loader, finetune_datamodule.train_dataloader(),
               teacher_model, student_model, avg_student_model, criterion,
               teacher_optimizer, student_optimizer, teacher_scheduler, student_scheduler, teacher_scaler, student_scaler)
    return
    

if __name__ == "__main__":
    main()