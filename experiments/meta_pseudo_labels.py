from experiments.common import seed_all
from typing import *
import timm
import pandas as pd
from timm.scheduler.cosine_lr import CosineLRScheduler
from collections import OrderedDict
from functools import partial

import logging
import time
import os

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
from experiments import layer_wise_learning_rate_decay, save_checkpoint, model_load_state_dict, get_lr, AverageMeter, get_lr_scheduler
import cv2

logger = logging.getLogger(__name__)

hyperparams = {
    "project_name": "AIProjectMetaPseudoLabels",
    "entity_name": "kn-bmi",
    "seed": 42,
    # "t_initial": 30,
    # "eta_min": 1e-7,
    # "warmup_lr_init": 1e-5,
    "warmup_steps": 5000,
    # "cycle_mul": 1,
    # "cycle_decay": 0.1,
    "lr": 1e-4,
    "batch_size_labeled": 16,
    "batch_size_unlabeled": 128,
    "lr_decay": 0.4, # Layer decay should be fine for both student and teacher 
                     #(probably if applied then with same parameter value to both), but it's up to further consideration
    "weight_decay": 5e-3,
    "betas": (0.9, 0.999),
    "nesterov": True,
    "weights": torch.Tensor([0.75, 1, 1, 1]),
    "workers": 1,
    "world_size": 1, # World size corresponds to number of processes (usually one per GPU)
    "start_step": 0,
    "total_steps": 25000,
    "eval_step": 1000,
    "local_rank": -1, # Local rank corresponds to process id within a node (world size / local rank explanation: 
                      # https://stackoverflow.com/questions/58271635/in-distributed-computing-what-are-world-size-and-rank)
    "gpu": 0,
    "device": torch.device('cuda', 0),
    "amp": True,
    "uda_temperature": 0.8,
    "threshold": 0.95,
    "uda_factor": 2.5,
    "uda_steps": 10000,
    "grad_clip": 0, # Check it out
    "ema": False, # May try.
    "best_accuracy": 0.,
    "best_f1_macro": 0.,
    "best_accuracy_teacher": 0.,
    "best_f1_macro_teacher": 0.,
    "save_path": r"C:\Users\Adam\Desktop\Studia\Psy Tabakowa\eye-disease\Eye-disease\checkpoints",
    "name": "regnet_y_3.2gf",
    "finetune_epochs": 10, # Would be cool to get early stopping working here
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


def reset_samplers_for_distributed_training(labeled_dataloader, unlabeled_dataloader):
    labeled_epoch = 0
    unlabeled_epoch = 0
    labeled_dataloader.sampler.set_epoch(labeled_epoch)
    unlabeled_dataloader.sampler.set_epoch(unlabeled_epoch)

class Loops:
    def __init__(self, labeled_dataloader, unlabeled_dataloader, val_dataloader, finetune_dataloader, teacher_model, 
               student_model, avg_student_model, criterion, teacher_optimizer, student_optimizer, 
               teacher_scheduler, student_scheduler, teacher_scaler, student_scaler):
        self.labeled_dataloader = labeled_dataloader
        self.unlabeled_dataloader = unlabeled_dataloader
        self.val_dataloader = val_dataloader
        self.finetune_dataloader = finetune_dataloader
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.avg_student_model = avg_student_model
        self.criterion = criterion
        self.teacher_optimizer = teacher_optimizer
        self.student_optimizer = student_optimizer
        self.teacher_scheduler = teacher_scheduler
        self.student_scheduler = student_scheduler
        self.teacher_scaler = teacher_scaler
        self.student_scaler = student_scaler
        self.progress_bar = tqdm(range(hyperparams["eval_step"]), disable=hyperparams["local_rank"] not in [-1, 0])
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        s_losses = AverageMeter()
        t_losses = AverageMeter()
        t_losses_l = AverageMeter()
        t_losses_u = AverageMeter()
        t_losses_mpl = AverageMeter()
        mean_mask = AverageMeter()
        normal_counts = AverageMeter()
        glaucoma_counts = AverageMeter()
        amd_counts = AverageMeter()
        dr_counts = AverageMeter()
        dot_products = AverageMeter()
    
    def train_loop(self):
        if hyperparams["world_size"] > 1:
            reset_samplers_for_distributed_training(self.labeled_loader, self.unlabeled_loader)
        

def train_loop(labeled_dataloader, unlabeled_dataloader, val_dataloader, finetune_dataloader, teacher_model, 
               student_model, avg_student_model, criterion, teacher_optimizer, student_optimizer, 
               teacher_scheduler, student_scheduler, teacher_scaler, student_scaler):
    """Mostly imported from https://github.com/kekmodel/MPL-pytorch/blob/main/main.py"""
    
    
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
    normal_counts = AverageMeter()
    glaucoma_counts = AverageMeter()
    amd_counts = AverageMeter()
    dr_counts = AverageMeter()
    dot_products = AverageMeter()
    
    for step in range(hyperparams["start_step"], hyperparams["total_steps"]):
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
            normal_counts = AverageMeter()
            glaucoma_counts = AverageMeter()
            amd_counts = AverageMeter()
            dr_counts = AverageMeter()
            dot_products = AverageMeter()

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
            # Original unlabeled images, augmented unlabeled images
            (images_uw, images_us), _ = unlabeled_iter.next()
        except:
            if hyperparams["world_size"] > 1:
                unlabeled_epoch += 1
                unlabeled_loader.sampler.set_epoch(unlabeled_epoch)
            unlabeled_iter = iter(unlabeled_loader)
            # Original unlabeled images, augmented unlabeled images
            (images_uw, images_us), _ = unlabeled_iter.next()
        
        data_time.update(time.time() - end)

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

            # Teacher labeled loss
            t_loss_l = criterion(t_logits_l, targets)

            # According to: https://github.com/google-research/google-research/issues/534#issuecomment-769526361
            # Soft labels and student training should be on augmented unlabeled data?
            # UDA loss seems to work like this: 
            # sample soft pseudo labels from original unlabeled data -> sample logits from augmented unlabeled data ->
            # -> for enough confident predictions on original labeled data calculate average cross entropy loss
            # between soft pseudo labels and logits from augmented unlabeled data ->
            # scale that loss by some factor and add loss on batch of labeled data. Unlabeled data should have ~7,8x larger batch size. 
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
            # That's probably wrong. Student probably should be trained on augmented data (as is) but with hard pseudo labels of teacher augmented data logits
            # as pointed here: https://github.com/google-research/google-research/issues/534#issuecomment-769526361.
            # and here: https://github.com/google-research/google-research/issues/534#issuecomment-769527910
            # s_loss = criterion(s_logits_us, hard_pseudo_label)
            # My (possibly) correct implementation.
            _, hard_pseudo_label_aug = torch.max(t_logits_us.detach(), dim=-1)
            s_loss = criterion(s_logits_us, hard_pseudo_label_aug)
            hard_pseudo_label_pd = pd.Series(hard_pseudo_label.cpu().numpy())
            label_stats = hard_pseudo_label_pd.value_counts()
            if 0 in label_stats.keys():
                normal_counts.update(label_stats[0])
            if 1 in label_stats.keys():
                glaucoma_counts.update(label_stats[1])
            if 2 in label_stats.keys():
                amd_counts.update(label_stats[2])
            if 3 in label_stats.keys():
                dr_counts.update(label_stats[3])

        s_scaler.scale(s_loss).backward()
        if hyperparams["grad_clip"] > 0:
            s_scaler.unscale_(s_optimizer)
            nn.utils.clip_grad_norm_(student_model.parameters(), hyperparams["grad_clip"])
        s_scaler.step(s_optimizer)
        s_scaler.update()
        s_scheduler.step()
        if hyperparams["ema"] > 0:
            avg_student_model.update_parameters(student_model)

        with amp.autocast(enabled=hyperparams["amp"]):
            with torch.no_grad():
                s_logits_l = student_model(images_l)
            s_loss_l_new = F.cross_entropy(s_logits_l.detach(), targets)

            # theoretically correct formula (https://github.com/kekmodel/MPL-pytorch/issues/6)
            # dot_product = s_loss_l_old - s_loss_l_new

            # author's code formula
            # As pointed out by https://github.com/kekmodel/MPL-pytorch/issues/6#issuecomment-851636630 the way the dot product is 
            # calculated doesn't seem to matter that much. Moreover as denoted in the issue changing to theoretically correct formula may actually hurt performance.
            dot_product = s_loss_l_new - s_loss_l_old # We don't use cosine similarity, because we use Taylor approximation.
            # moving_dot_product = moving_dot_product * 0.99 + dot_product * 0.01
            # dot_product = dot_product - moving_dot_product

            # Teacher MPL loss is calculated on hard labels from augmented unlabeled data and logits from augmented unlabeled data.
            _, hard_pseudo_label = torch.max(t_logits_us.detach(), dim=-1)
            t_loss_mpl = dot_product * F.cross_entropy(t_logits_us, hard_pseudo_label)
            t_loss = t_loss_uda + t_loss_mpl

        t_scaler.scale(t_loss).backward()
        if hyperparams["grad_clip"] > 0:
            t_scaler.unscale_(t_optimizer)
            nn.utils.clip_grad_norm_(teacher_model.parameters(), hyperparams["grad_clip"])
        t_scaler.step(t_optimizer)
        t_scaler.update()
        t_scheduler.step()

        teacher_model.zero_grad()
        student_model.zero_grad()

        if hyperparams["world_size"] > 1:
            s_loss = reduce_tensor(s_loss.detach(), hyperparams["world_size"])
            t_loss = reduce_tensor(t_loss.detach(), hyperparams["world_size"])
            t_loss_l = reduce_tensor(t_loss_l.detach(), hyperparams["world_size"])
            t_loss_u = reduce_tensor(t_loss_u.detach(), hyperparams["world_size"])
            t_loss_mpl = reduce_tensor(t_loss_mpl.detach(), hyperparams["world_size"])
            dot_product = reduce_tensor(dot_product.detach(), hyperparams["world_size"])
            mask = reduce_tensor(mask, hyperparams["world_size"])

        s_losses.update(s_loss.item())
        t_losses.update(t_loss.item())
        t_losses_l.update(t_loss_l.item())
        t_losses_u.update(t_loss_u.item())
        t_losses_mpl.update(t_loss_mpl.item())
        dot_products.update(dot_product.item())
        mean_mask.update(mask.mean().item())

        batch_time.update(time.time() - end)
        pbar.set_description(
            f"Train Iter: {step+1:3}/{hyperparams['world_size']:3}. "
            f"LR: {get_lr(t_optimizer):.4f}. Data: {data_time.avg:.2f}s. "
            f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. "
            f"T_Loss: {t_losses.avg:.4f}. Mask: {mean_mask.avg:.4f}. "
            f"Normal: {normal_counts.avg:.2f}. Glaucoma: {glaucoma_counts.avg:.2f}. "
            f"AMD: {amd_counts.avg:.2f}. DR: {dr_counts.avg:.2f}. ")
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
                           "train_uda_mask_mean": mean_mask.avg,
                           "normal_pseudo_avg_per_batch": normal_counts.avg,
                           "glaucoma_pseudo_avg_per_batch": amd_counts.avg,
                           "amd_pseudo_avg_per_batch": glaucoma_counts.avg,
                           "dr_pseudo_avg_per_batch": dr_counts.avg,
                           "normal_pseudo_sum": normal_counts.sum,
                           "glaucoma_pseudo_sum": amd_counts.sum,
                           "amd_pseudo_sum": glaucoma_counts.sum,
                           "dr_pseudo_sum": dr_counts.sum})

                val_model = avg_student_model if avg_student_model is not None else student_model
                val_loss, acc, f1_macro = evaluate(val_loader, val_model, criterion)

                wandb.log({"val_loss": val_loss,
                           "val_accuracy": acc,
                           "val_f1_macro": f1_macro})

                is_best = acc > hyperparams["best_f1_macro"]
                if is_best:
                    hyperparams["best_f1_macro"] = f1_macro
                    hyperparams["best_accuracy"] = acc

                logger.info(f"f1_macro: {f1_macro:.2f}")
                logger.info(f"Best f1_macro: {hyperparams['best_f1_macro']:.2f}")
                
                val_loss_teacher, acc_teacher, f1_macro_teacher = evaluate(val_loader, teacher_model, criterion)

                wandb.log({"val_loss_teacher": val_loss_teacher,
                           "val_accuracy_teacher": acc_teacher,
                           "val_f1_macro_teacher": f1_macro_teacher})

                is_best_teacher = acc_teacher > hyperparams["best_f1_macro_teacher"]
                if is_best_teacher:
                    hyperparams["best_f1_macro_teacher"] = f1_macro_teacher
                    hyperparams["best_accuracy_teacher"] = acc_teacher

                logger.info(f"f1_macro_teacher: {f1_macro_teacher:.2f}")
                logger.info(f"Best f1_macro_teacher: {hyperparams['best_f1_macro_teacher']:.2f}")

                save_checkpoint({
                    'step': step + 1,
                    'teacher_state_dict': teacher_model.state_dict(),
                    'student_state_dict': student_model.state_dict(),
                    'avg_state_dict': avg_student_model.state_dict() if avg_student_model is not None else None,
                    'accuracy': hyperparams["best_accuracy"],
                    'f1_macro': hyperparams["best_f1_macro"],
                    'teacher_optimizer': t_optimizer.state_dict(),
                    'student_optimizer': s_optimizer.state_dict(),
                    'teacher_scheduler': t_scheduler.state_dict(),
                    'student_scheduler': s_scheduler.state_dict(),
                    'teacher_scaler': t_scaler.state_dict(),
                    'student_scaler': s_scaler.state_dict(),
                }, is_best)

    if hyperparams["local_rank"] in [-1, 0]:
        wandb.log({"best_val_acc": hyperparams["best_accuracy"]})
        wandb.log({"best_f1_macro": hyperparams['best_f1_macro']})
        wandb.log({"best_val_acc_teacher": hyperparams["best_accuracy_teacher"]})
        wandb.log({"best_f1_macro_teacher": hyperparams['best_f1_macro_teacher']})
    

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