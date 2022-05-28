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
from utils.metrics import f1_score, sensitivity, specificity, roc_auc, precision
from torchmetrics.functional import accuracy
from torchmetrics.functional import f1_score as f1
from adamp import AdamP
from dataset import EyeDiseaseDataModule
from dataset.transforms import train_transforms, test_val_transforms, mpl_transforms
from dataset.resamplers import identity_resampler
from methods import RegNetY3_2gf, RegNetY400MF
from experiments import layer_wise_learning_rate_decay, save_checkpoint, model_load_state_dict, get_lr, AverageMeter, get_lr_scheduler, reduce_tensor
import cv2
from common import load_lightning_model

logger = logging.getLogger(__name__)

# TODO: Test usual train transforms instead of RandAugment transforms, because they are confirmed to be label preserving
# RandAugment is usually suitable for coarse-grained image classification
hyperparams = {
    "project_name": "AIProjectMetaPseudoLabels",
    "entity_name": "kn-bmi",
    "seed": 42,
    "warmup_steps": 6000, 
    "lr": 1e-4,
    "batch_size_labeled": 8,
    "batch_size_unlabeled": 64, # Should be ~8x batch_size_labeled
    "lr_decay": 0.9, # Checkout if layer decay actually improves performance
    "weight_decay": 5e-3, # Probably necessary, otherwise seems to overfit to teacher's destillation
    "betas": (0.9, 0.999),
    "nesterov": True,
    "weights": torch.Tensor([1, 1, 1, 1]), # For MPL stage
    "workers": 2,
    "world_size": 1, # World size corresponds to number of processes (usually one per GPU)
    "start_step": 0,
    "total_steps": 30000, # Around 10 epochs, already seems infeasible for kaggle, should be 10-30x more
    "eval_step": 500,
    "local_rank": -1, # Local rank corresponds to process id within a node (world size / local rank explanation: 
                      # https://stackoverflow.com/questions/58271635/in-distributed-computing-what-are-world-size-and-rank)
    "gpu": 0, 
    "device": torch.device('cuda', 0),
    "amp": True,
    "uda_temperature": 0.8, # Should be tuned
    "threshold": 0.95, # Should be tuned
    "uda_factor": 1.0, # According to paper should rather be > 1, although uda loss seems to be pretty large in eye disease problem
    "mpl_weight": 10.0, # Does not exist in original paper, although mpl loss is very little compared to uda loss
    "uda_steps": 30000, # Number of steps after which uda loss weight achieves uda_factor value (TODO: confirm with paper)
    "grad_clip": 2, # Should be tuned
    "ema": False, # Probably beneficial for evaluation, although unfeasible
    "best_accuracy": 0.,
    "best_f1_macro": 0.,
    "best_accuracy_teacher": 0.,
    "best_f1_macro_teacher": 0.,
    "best_finetune_f1_macro": 0.,
    "best_finetune_accuracy": 0.,
    "save_path": r"./checkpoints/",
    "name": "regnet_y_3.2gf",
    "train_split_name": "train",
    "val_split_name": "val",
    "test_split_name": "test",
    "unlabeled_split_name": "pretrain",
    "normalize": True,
    "resampler": identity_resampler,
    "resume": False,
    "evaluate": False,
    "finetune": False,
    "finetune_weights": torch.Tensor([0.408, 0.408, 1, 0.408]), # Weights chosen as in upcoming paper
    "finetune_epochs": 15, # Would be cool to get early stopping working here
    "finetune_lr": 5e-5, # Lower learning rate for finetuning
    "finetune_weight_decay": 1e-8 # Typical weight decay value for finetuning with decoupled weight decay
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
        self.labeled_epoch = 0
        self.labeled_iter = None
        self.umlabeled_iter = None
        
    def reset_average_meters(self):
        self.progress_bar = tqdm(range(hyperparams["eval_step"]), disable=hyperparams["local_rank"] not in [-1, 0])
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.student_pseudo_losses = AverageMeter()
        self.teacher_losses = AverageMeter()
        self.teacher_losses_labeled = AverageMeter()
        self.teacher_losses_uda = AverageMeter()
        self.teacher_losses_mpl = AverageMeter()
        self.mean_mask = AverageMeter()
        self.dot_products = AverageMeter()
        self.normal_counts = AverageMeter()
        self.glaucoma_counts = AverageMeter()
        self.amd_counts = AverageMeter()
        self.dr_counts = AverageMeter()
        self.accuracy = AverageMeter()
        self.f1s = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
        self.sensitivities = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
        self.specificities = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
        self.precisions = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
        self.roc_aucs = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]    
        self.teacher_f1s = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
        self.teacher_sensitivities = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
        self.teacher_specificities = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
        self.teacher_precisions = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
        self.teacher_roc_aucs = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]    
    
    def get_labeled_batch(self):
        try:
            return self.labeled_dataloader.next()
        except:
            if hyperparams["world_size"] > 1:
                self.labeled_epoch += 1
                self.labeled_dataloader.sampler.set_epoch(self.labeled_epoch)
            self.labeled_iter = iter(self.labeled_dataloader)
            return self.labeled_iter.next()

    def get_unlabeled_batch(self):
        try:
            # Original unlabeled images, augmented unlabeled images
            return self.unlabeled_iter.next()
        except:
            if hyperparams["world_size"] > 1:
                self.unlabeled_epoch += 1
                self.unlabeled_dataloader.sampler.set_epoch(self.unlabeled_epoch)
            self.unlabeled_iter = iter(self.unlabeled_dataloader)
            # Original unlabeled images, augmented unlabeled images
            return self.unlabeled_iter.next()
    
    def get_teacher_logits(self, images_labeled, images_unlabeled_orig, images_unlabeled_aug):
        batch_size = images_labeled.shape[0]
        teacher_images = torch.cat((images_labeled, images_unlabeled_orig, images_unlabeled_aug))
        teacher_logits = self.teacher_model(teacher_images)
        teacher_logits_labeled = teacher_logits[:batch_size]
        teacher_logits_unlabeled_orig, teacher_logits_unlabeled_aug = teacher_logits[batch_size:].chunk(2)
        del teacher_logits
        return teacher_logits_labeled, teacher_logits_unlabeled_orig, teacher_logits_unlabeled_aug

    def get_student_logits(self, images_labeled, images_unlabeled_aug):
        batch_size = images_labeled.shape[0]
        student_images = torch.cat((images_labeled, images_unlabeled_aug))
        student_logits = self.student_model(student_images)
        student_logits_labeled = student_logits[:batch_size]
        student_logits_unlabeled_aug = student_logits[batch_size:]
        del student_logits
        return student_logits_labeled, student_logits_unlabeled_aug
    
    def get_teacher_losses(self, step, teacher_logits_labeled, teacher_logits_unlabeled_orig, teacher_logits_unlabeled_aug, targets):
        teacher_loss_labeled = self.criterion(teacher_logits_labeled, targets)

        # According to: https://github.com/google-research/google-research/issues/534#issuecomment-769526361
        # Soft labels and student training should be on augmented unlabeled data
        # UDA loss seems to work like this (TODO: confirm with paper): 
        # sample soft pseudo labels from original unlabeled data -> sample logits from augmented unlabeled data ->
        # -> for enough confident predictions on original labeled data calculate average cross entropy loss
        # between soft pseudo labels and logits from augmented unlabeled data ->
        # scale that loss by some factor and add loss on batch of labeled data. Unlabeled data should have ~7,8x larger batch size. 
        soft_pseudo_labels = torch.softmax(teacher_logits_unlabeled_orig.detach() / hyperparams["uda_temperature"], dim=-1)
        max_probs, hard_pseudo_labels = torch.max(soft_pseudo_labels, dim=-1)
        mask = max_probs.ge(hyperparams["threshold"]).float()
        teacher_loss_unlabeled_aug = torch.mean(
            -(soft_pseudo_labels * torch.log_softmax(teacher_logits_unlabeled_aug, dim=-1)).sum(dim=-1) * mask
        )
        uda_weight = hyperparams["uda_factor"] * min(1., (step + 1) / hyperparams["uda_steps"])
        teacher_loss_uda = teacher_loss_labeled + uda_weight * teacher_loss_unlabeled_aug
        return teacher_loss_labeled, teacher_loss_uda, hard_pseudo_labels, mask
    
    def get_student_losses(self, student_logits_labeled, student_logits_unlabeled_aug, targets, teacher_logits_unlabeled_aug):
        student_loss_labeled_old = F.cross_entropy(student_logits_labeled.detach(), targets)
        # That's probably wrong. Student probably should be trained on augmented data (as is) 
        # but with hard pseudo labels of teacher augmented data logits
        # as pointed here: https://github.com/google-research/google-research/issues/534#issuecomment-769526361.
        # and here: https://github.com/google-research/google-research/issues/534#issuecomment-769527910
        # s_loss = criterion(s_logits_us, hard_pseudo_label)
        # My (possibly) correct implementation:
        soft_pseudo_label_aug = torch.softmax(teacher_logits_unlabeled_aug.detach(), dim=-1)
        _, hard_pseudo_label_aug = torch.max(soft_pseudo_label_aug, dim=-1)
        student_loss = self.criterion(student_logits_unlabeled_aug, hard_pseudo_label_aug)
        return student_loss_labeled_old, student_loss
    
    def update_pseudo_labeling_stats(self, hard_pseudo_labels):
        hard_pseudo_label_pd = pd.Series(hard_pseudo_labels.cpu().numpy())
        label_stats = hard_pseudo_label_pd.value_counts()
        if 0 in label_stats.keys():
            self.normal_counts.update(label_stats[0])
        if 1 in label_stats.keys():
            self.glaucoma_counts.update(label_stats[1])
        if 2 in label_stats.keys():
            self.amd_counts.update(label_stats[2])
        if 3 in label_stats.keys():
            self.dr_counts.update(label_stats[3])
            
    def update_student(self, student_pseudo_loss):
        self.student_scaler.scale(student_pseudo_loss).backward()
        if hyperparams["grad_clip"] > 0:
            self.student_scaler.unscale_(self.student_optimizer)
            nn.utils.clip_grad_norm_(self.student_model.parameters(), hyperparams["grad_clip"])
        self.student_scaler.step(self.student_optimizer)
        self.student_scaler.update()
        self.student_scheduler.step()
        if hyperparams["ema"] > 0:
            self.avg_student_model.update_parameters(self.student_model)
    
    def get_teacher_final_loss(self, images_labeled, targets, student_loss_labeled_old, teacher_logits_unlabeled_aug, teacher_loss_uda):
        with amp.autocast(enabled=hyperparams["amp"]):
            with torch.no_grad():
                student_logits_labeled_new = self.student_model(images_labeled)
            student_loss_labeled_new = F.cross_entropy(student_logits_labeled_new.detach(), targets)

            # theoretically correct formula (https://github.com/kekmodel/MPL-pytorch/issues/6)
            # dot_product = s_loss_l_old - s_loss_l_new

            # author's code formula
            # As pointed out by https://github.com/kekmodel/MPL-pytorch/issues/6#issuecomment-851636630 the way the dot product is 
            # calculated doesn't seem to matter that much. Moreover as denoted in the issue changing to theoretically correct formula may actually hurt performance.
            dot_product = student_loss_labeled_new - student_loss_labeled_old # We don't use cosine similarity, because we use Taylor approximation.
            # moving_dot_product = moving_dot_product * 0.99 + dot_product * 0.01
            # dot_product = dot_product - moving_dot_product

            # Teacher MPL loss is calculated on hard labels from augmented unlabeled data and logits from augmented unlabeled data.
            _, hard_pseudo_labels_aug = torch.max(teacher_logits_unlabeled_aug.detach(), dim=-1)
            # MPL loss takes low values compared to uda loss
            t_loss_mpl = dot_product * F.cross_entropy(teacher_logits_unlabeled_aug, hard_pseudo_labels_aug)
            teacher_final_loss = teacher_loss_uda + hyperparams["mpl_weight"] * t_loss_mpl
            return teacher_final_loss, hyperparams["mpl_weight"] * t_loss_mpl, dot_product
    
    def update_teacher(self, teacher_loss):
        self.teacher_scaler.scale(teacher_loss).backward()
        if hyperparams["grad_clip"] > 0:
            self.teacher_scaler.unscale_(self.teacher_optimizer)
            nn.utils.clip_grad_norm_(self.teacher_model.parameters(), hyperparams["grad_clip"])
        self.teacher_scaler.step(self.teacher_optimizer)
        self.teacher_scaler.update()
        self.teacher_scheduler.step()
    
    def reduce_tensors(self, student_pseudo_loss, teacher_loss, teacher_loss_labeled, teacher_loss_uda, teacher_loss_mpl, dot_product, mask):
        if hyperparams["world_size"] > 1:
            student_pseudo_loss = reduce_tensor(student_pseudo_loss.detach(), hyperparams["world_size"])
            teacher_loss = reduce_tensor(teacher_loss.detach(), hyperparams["world_size"])
            teacher_loss_labeled = reduce_tensor(teacher_loss_labeled.detach(), hyperparams["world_size"])
            teacher_loss_uda = reduce_tensor(teacher_loss_uda.detach(), hyperparams["world_size"])
            teacher_loss_mpl = reduce_tensor(teacher_loss_mpl.detach(), hyperparams["world_size"])
            dot_product = reduce_tensor(dot_product.detach(), hyperparams["world_size"])
            mask = reduce_tensor(mask, hyperparams["world_size"])
        return student_pseudo_loss, teacher_loss, teacher_loss_labeled, teacher_loss_uda, teacher_loss_mpl, dot_product, mask
    
    def update_average_losses(self, student_pseudo_loss, teacher_loss, teacher_loss_labeled, teacher_loss_uda, teacher_loss_mpl, dot_product, mask):
        self.student_pseudo_losses.update(student_pseudo_loss.item())
        self.teacher_losses.update(teacher_loss.item())
        self.teacher_losses_labeled.update(teacher_loss_labeled.item())
        self.teacher_losses_uda.update(teacher_loss_uda.item())
        self.teacher_losses_mpl.update(teacher_loss_mpl.item())
        self.dot_products.update(dot_product.item())
        self.mean_mask.update(mask.mean().item())
    
    def update_progress_bar(self, step, end):
        self.batch_time.update(time.time() - end)
        self.progress_bar.set_description(
            f"Train Iter: {step+1:3}/{hyperparams['world_size']:3}. "
            f"LR: {get_lr(self.teacher_optimizer):.4f}. Data: {self.data_time.avg:.2f}s. "
            f"Batch: {self.batch_time.avg:.2f}s. S_Loss: {self.student_pseudo_losses.avg:.4f}. "
            f"T_Loss: {self.teacher_losses.avg:.4f}. Mask: {self.mean_mask.avg:.4f}. "
            f"Normal: {self.normal_counts.avg:.2f}. Glaucoma: {self.glaucoma_counts.avg:.2f}. "
            f"AMD: {self.amd_counts.avg:.2f}. DR: {self.dr_counts.avg:.2f}. ")
        self.progress_bar.update()
        if hyperparams["local_rank"] in [-1, 0]:
            wandb.log({"lr": hyperparams["lr"]})
    
    def log_metrics(self):
        wandb.log({
            "val_student_f1_normal": self.f1s[0].avg,
            "val_student_f1_glaucoma": self.f1s[1].avg,
            "val_student_f1_amd": self.f1s[2].avg,
            "val_student_f1_dr": self.f1s[3].avg,
            "val_student_sensitivity_normal": self.sensitivities[0].avg,
            "val_student_sensitivity_glaucoma": self.sensitivities[1].avg,
            "val_student_sensitivity_amd": self.sensitivities[2].avg,
            "val_student_sensitivity_dr": self.sensitivities[3].avg,
            "val_student_specificity_normal": self.specificities[0].avg,
            "val_student_specificity_glaucoma": self.specificities[1].avg,
            "val_student_specificity_amd": self.specificities[2].avg,
            "val_student_specificity_dr": self.specificities[3].avg,
            "val_student_precision_normal": self.precisions[0].avg,
            "val_student_precision_glaucoma": self.precisions[1].avg,
            "val_student_precision_amd": self.precisions[2].avg,
            "val_student_precision_dr": self.precisions[3].avg,
            "val_student_roc_auc_normal": self.roc_aucs[0].avg,
            "val_student_roc_auc_glaucoma": self.roc_aucs[1].avg,
            "val_student_roc_auc_amd": self.roc_aucs[2].avg,
            "val_student_roc_auc_dr": self.roc_aucs[3].avg
        })
        wandb.log({
            "val_teacher_f1_normal": self.teacher_f1s[0].avg,
            "val_teacher_f1_glaucoma": self.teacher_f1s[1].avg,
            "val_teacher_f1_amd": self.teacher_f1s[2].avg,
            "val_teacher_f1_dr": self.teacher_f1s[3].avg,
            "val_teacher_sensitivity_normal": self.teacher_sensitivities[0].avg,
            "val_teacher_sensitivity_glaucoma": self.teacher_sensitivities[1].avg,
            "val_teacher_sensitivity_amd": self.teacher_sensitivities[2].avg,
            "val_teacher_sensitivity_dr": self.teacher_sensitivities[3].avg,
            "val_teacher_specificity_normal": self.teacher_specificities[0].avg,
            "val_teacher_specificity_glaucoma": self.teacher_specificities[1].avg,
            "val_teacher_specificity_amd": self.teacher_specificities[2].avg,
            "val_teacher_specificity_dr": self.teacher_specificities[3].avg,
            "val_teacher_precision_normal": self.teacher_precisions[0].avg,
            "val_teacher_precision_glaucoma": self.teacher_precisions[1].avg,
            "val_teacher_precision_amd": self.teacher_precisions[2].avg,
            "val_teacher_precision_dr": self.teacher_precisions[3].avg,
            "val_teacher_roc_auc_normal": self.teacher_roc_aucs[0].avg,
            "val_teacher_roc_auc_glaucoma": self.teacher_roc_aucs[1].avg,
            "val_teacher_roc_auc_amd": self.teacher_roc_aucs[2].avg,
            "val_teacher_roc_auc_dr": self.teacher_roc_aucs[3].avg
        })
    
    def training_closure(self, step):
        wandb.log({"train_student_loss": self.student_pseudo_losses.avg,
                    "train_teacher_loss": self.teacher_losses.avg,
                    "train_teacher_labeled_loss": self.teacher_losses_labeled.avg,
                    "train_teacher_uda_loss": self.teacher_losses_uda.avg,
                    "train_teacher_mpl_loss": self.teacher_losses_mpl.avg,
                    "train_uda_mask_mean": self.mean_mask.avg,
                    "normal_pseudo_avg_per_batch": self.normal_counts.avg,
                    "glaucoma_pseudo_avg_per_batch": self.amd_counts.avg,
                    "amd_pseudo_avg_per_batch": self.glaucoma_counts.avg,
                    "dr_pseudo_avg_per_batch": self.dr_counts.avg,
                    "normal_pseudo_sum": self.normal_counts.sum,
                    "glaucoma_pseudo_sum": self.amd_counts.sum,
                    "amd_pseudo_sum": self.glaucoma_counts.sum,
                    "dr_pseudo_sum": self.dr_counts.sum})

        val_model = self.avg_student_model if self.avg_student_model is not None else self.student_model
        val_loss, val_acc, val_f1 = self.evaluate(val_model, is_student=True)
        teacher_val_loss, teacher_val_acc, teacher_val_f1 = self.evaluate(self.teacher_model, is_student=False)

        wandb.log({"val_loss": val_loss,
                "val_student_accuracy": val_acc,
                "val_student_f1_macro": val_f1})
        
        wandb.log({"val_loss": teacher_val_loss,
                "val_teacher_accuracy": teacher_val_acc,
                "val_teacher_f1_macro": teacher_val_f1})
        
        is_best = val_f1 > hyperparams["best_f1_macro"]
        if is_best:
            hyperparams["best_f1_macro"] = val_f1
            hyperparams["best_accuracy"] = val_acc
        
        is_best = teacher_val_f1 > hyperparams["best_f1_macro_teacher"]
        if is_best:
            hyperparams["best_f1_macro_teacher"] = teacher_val_f1
            hyperparams["best_accuracy_teacher"] = teacher_val_acc
        
        self.log_metrics()

        save_checkpoint({
            'step': step + 1,
            'teacher_state_dict': self.teacher_model.state_dict(),
            'student_state_dict': self.student_model.state_dict(),
            'avg_state_dict': self.avg_student_model.state_dict() if self.avg_student_model is not None else None,
            'teacher_optimizer': self.teacher_optimizer.state_dict(),
            'student_optimizer': self.student_optimizer.state_dict(),
            'teacher_scheduler': self.teacher_scheduler.state_dict(),
            'student_scheduler': self.student_scheduler.state_dict(),
            'teacher_scaler': self.teacher_scaler.state_dict(),
            'student_scaler': self.student_scaler.state_dict(),
        }, is_best)
    
    def evaluate(self, model, is_student):
        losses = AverageMeter()
        acc_meter = AverageMeter()
        f1_macro_meter = AverageMeter()
        model.eval()
        val_iter = tqdm(self.val_dataloader, disable=hyperparams["local_rank"] not in [-1, 0])
        with torch.no_grad():
            for step, (images, targets) in enumerate(val_iter):
                batch_size = images.shape[0]
                images = images.to(hyperparams["device"])
                targets = targets.to(hyperparams["device"])
                with amp.autocast(enabled=hyperparams["amp"]):
                    outputs = model(images)
                    loss = self.criterion(outputs, targets)

                acc_res = accuracy(outputs, targets)
                f1_macro = f1(outputs, targets, average="macro", num_classes=4)
                if is_student:
                    for i in range(len(self.f1s)):
                        f1_res = f1_score(outputs, targets, current_class=i)
                        self.f1s[i].update(f1_res)
                        
                    for i in range(len(self.sensitivities)):
                        sens_res = sensitivity(outputs, targets, current_class=i)
                        self.sensitivities[i].update(sens_res)
                        
                    for i in range(len(self.specificities)):
                        spec_res = specificity(outputs, targets, current_class=i)
                        self.specificities[i].update(spec_res)
                        
                    for i in range(len(self.precisions)):
                        prec_res = precision(outputs, targets, current_class=i)
                        self.precisions[i].update(prec_res)
                    
                    for i in range(len(self.roc_aucs)):
                        roc_res = roc_auc(outputs, targets, current_class=i)
                        self.roc_aucs[i].update(roc_res)
                else:
                    for i in range(len(self.f1s)):
                        f1_res = f1_score(outputs, targets, current_class=i)
                        self.teacher_f1s[i].update(f1_res)
                        
                    for i in range(len(self.sensitivities)):
                        sens_res = sensitivity(outputs, targets, current_class=i)
                        self.teacher_sensitivities[i].update(sens_res)
                        
                    for i in range(len(self.specificities)):
                        spec_res = specificity(outputs, targets, current_class=i)
                        self.teacher_specificities[i].update(spec_res)
                        
                    for i in range(len(self.precisions)):
                        prec_res = precision(outputs, targets, current_class=i)
                        self.teacher_precisions[i].update(prec_res)
                    
                    for i in range(len(self.roc_aucs)):
                        roc_res = roc_auc(outputs, targets, current_class=i)
                        self.teacher_roc_aucs[i].update(roc_res)
                losses.update(loss.item(), batch_size)
                acc_meter.update(acc_res, batch_size)
                f1_macro_meter.update(f1_macro, batch_size)
                val_iter.set_description(
                    f"Val Iter: {step+1:3}/{len(self.val_dataloader):3}."
                    f"Loss: {losses.avg:.4f}. "
                    f"accuracy: {acc_meter.avg:.2f}. f1_macro: {f1_macro_meter.avg:.2f}. ")

        val_iter.close()
        return losses.avg, acc_meter.avg, f1_macro_meter.avg
    
    def finetune(self):
        self.reset_average_meters()
        parameters = layer_wise_learning_rate_decay(self.student_model)
        optimizer = AdamP(
            parameters,
            lr=hyperparams["finetune_lr"],
            betas=(hyperparams["betas"][0], hyperparams["betas"][1]),
            weight_decay=hyperparams["finetune_weight_decay"],
            nesterov=hyperparams["nesterov"],
        )
        scheduler = get_lr_scheduler(optimizer)
        scaler = amp.GradScaler(enabled=hyperparams["amp"])

        logger.info("***** Running Finetuning *****")
        logger.info(f"   Finetuning steps = {len(self.labeled_dataloader)* hyperparams['finetune_epochs']}")

        for epoch in range(hyperparams['finetune_epochs']):
            if hyperparams['world_size'] > 1:
                self.labeled_dataloader.sampler.set_epoch(epoch + 624)

            self.losses = AverageMeter()
            self.student_model.train()
            end = time.time()
            labeled_iter = tqdm(self.labeled_dataloader, disable=hyperparams['local_rank'] not in [-1, 0])
            for step, (images, targets) in enumerate(labeled_iter):
                self.data_time.update(time.time() - end)
                batch_size = images.shape[0]
                images = images.to(hyperparams['device'])
                targets = targets.to(hyperparams['device'])
                with amp.autocast(enabled=hyperparams['amp']):
                    self.student_model.zero_grad()
                    outputs = self.student_model(images)
                    loss = self.criterion(outputs, targets)
                        
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                if hyperparams['world_size'] > 1:
                    loss = reduce_tensor(loss.detach(), hyperparams['world_size'])
                self.losses.update(loss.item(), batch_size)
                self.batch_time.update(time.time() - end)
                labeled_iter.set_description(
                    f"Finetune Epoch: {epoch+1:2}/{hyperparams['finetune_epochs']:2}. Data: {self.data_time.avg:.2f}s. "
                    f"Batch: {self.batch_time.avg:.2f}s. Loss: {self.losses.avg:.4f}. ")
            labeled_iter.close()
            if hyperparams['local_rank'] in [-1, 0]:
                val_loss, val_acc, val_f1_macro = self.evaluate(self.student_model, is_student=True)
                wandb.log({"finetune_train_loss": self.losses.avg,
                        "finetune_val_loss": val_loss,
                        "finetune_accuracy": val_acc,
                        "finetune_f1_macro": val_f1_macro})

                is_best = val_f1_macro > hyperparams["best_finetune_f1_macro"]
                if is_best:
                    hyperparams["best_finetune_f1_macro"] = val_f1_macro
                    hyperparams["best_finetune_accuracy"] = val_acc

                logger.info(f"f1_macro: {val_f1_macro:.2f}")
                logger.info(f"Best f1_macro: {hyperparams['best_finetune_f1_macro']:.2f}")

                save_checkpoint({
                    'step': step + 1,
                    'best_accuracy': hyperparams['best_finetune_accuracy'],
                    'best_f1_macro': hyperparams['best_finetune_f1_macro'],
                    'student_state_dict': self.student_model.state_dict(),
                    'avg_state_dict': None,
                    'student_optimizer': optimizer.state_dict(),
                }, is_best, finetune=True)
            if hyperparams['local_rank'] in [-1, 0]:
                wandb.log({"best_finetune_f1_macro": hyperparams['best_finetune_f1_macro']})
                wandb.log({"best_finetune_accuracy": hyperparams['best_finetune_accuracy']})
        return
       
    def training_step(self, step):
        if step % hyperparams["eval_step"] == 0:
            self.reset_average_meters()

        self.teacher_model.train()
        self.student_model.train()
        end = time.time()

        images_labeled, targets = self.get_labeled_batch()
        (images_unlabeled_orig, images_unlabeled_aug), _ = self.get_unlabeled_batch()
        
        self.data_time.update(time.time() - end)

        images_labeled = images_labeled.to(hyperparams["device"])
        images_unlabeled_orig = images_unlabeled_orig.to(hyperparams["device"])
        images_unlabeled_aug = images_unlabeled_aug.to(hyperparams["device"])
        targets = targets.to(hyperparams["device"])
        with amp.autocast(enabled=hyperparams["amp"]):
            teacher_logits_labeled, teacher_logits_unlabeled_orig, teacher_logits_unlabeled_aug = self.get_teacher_logits(
                images_labeled,
                images_unlabeled_orig,
                images_unlabeled_aug
            )
            teacher_loss_labeled, teacher_loss_uda, hard_pseudo_labels, mask = self.get_teacher_losses(
                step,
                teacher_logits_labeled,
                teacher_logits_unlabeled_orig,
                teacher_logits_unlabeled_aug,
                targets
            )
            student_logits_labeled_old, student_logits_unlabeled_aug = self.get_student_logits(images_labeled, images_unlabeled_aug)
            student_loss_labeled_old, student_pseudo_loss = self.get_student_losses(
                student_logits_labeled_old,
                student_logits_unlabeled_aug,
                targets, 
                teacher_logits_unlabeled_aug
            )
            self.update_pseudo_labeling_stats(hard_pseudo_labels)

        self.update_student(student_pseudo_loss)        
        teacher_loss, teacher_loss_mpl, dot_product = self.get_teacher_final_loss(images_labeled, targets, 
                                                                                  student_loss_labeled_old, teacher_logits_unlabeled_aug, teacher_loss_uda)
        self.update_teacher(teacher_loss)
        self.teacher_model.zero_grad()
        self.student_model.zero_grad()

        student_pseudo_loss, teacher_loss, teacher_loss_labeled, teacher_loss_uda, teacher_loss_mpl, dot_product, mask = self.reduce_tensors(
            student_pseudo_loss, 
            teacher_loss,
            teacher_loss_labeled,
            teacher_loss_uda,
            teacher_loss_mpl,
            dot_product, mask
        )
        self.update_average_losses(student_pseudo_loss, teacher_loss, teacher_loss_labeled, teacher_loss_uda, teacher_loss_mpl, dot_product, mask)        
        self.update_progress_bar(step, end)

        hyperparams["num_eval"] = step // hyperparams["eval_step"]
        if (step + 1) % hyperparams["eval_step"] == 0:
            self.progress_bar.close()
            if hyperparams["local_rank"] in [-1, 0]:
                self.training_closure(step)
    
    def train_loop(self):
        if hyperparams["world_size"] > 1:
            reset_samplers_for_distributed_training(self.labeled_dataloader, self.unlabeled_dataloader)
        
        self.labeled_iter, self.unlabeled_iter = iter(self.labeled_dataloader), iter(self.unlabeled_dataloader)
        self.reset_average_meters()
        for step in range(hyperparams["start_step"], hyperparams["total_steps"]):
            self.training_step(step)
        
        ckpt_name = f'{hyperparams["save_path"]}/{hyperparams["name"]}_best.pth.tar'
        loc = f'cuda:{hyperparams["gpu"]}'
        checkpoint = torch.load(ckpt_name, map_location=loc)
        logger.info(f"=> loading checkpoint '{ckpt_name}'")
        if checkpoint['avg_state_dict'] is not None:
            model_load_state_dict(self.avg_student_model, checkpoint['avg_state_dict'])
        else:
            model_load_state_dict(self.student_model, checkpoint['student_state_dict'])
        self.finetune()
        return
    

def prepare_data_modules(teacher_model):
    labeled_datamodule = EyeDiseaseDataModule(
                csv_path=r'../input/corrected-data-splits/kaggle_pretrain_corrected_data_splits.csv',
                train_split_name=hyperparams["train_split_name"],
                val_split_name=hyperparams["val_split_name"],
                test_split_name=hyperparams["test_split_name"],
                train_transforms=test_val_transforms(teacher_model.input_size, hyperparams["normalize"], cv2.INTER_NEAREST),
                val_transforms=test_val_transforms(teacher_model.input_size, hyperparams["normalize"], cv2.INTER_NEAREST),
                test_transforms=test_val_transforms(teacher_model.input_size, hyperparams["normalize"], cv2.INTER_NEAREST),
                image_path_name='Path',
                target_name='Label',
                split_name='Split',
                batch_size=hyperparams["batch_size_labeled"],
                num_workers=hyperparams["workers"],
                shuffle_train=True,
                resampler=identity_resampler,
                pretraining=False,
                binary=False
    )
    labeled_datamodule.prepare_data()
    
    unlabeled_datamodule = EyeDiseaseDataModule(
                csv_path=r'../input/corrected-data-splits/kaggle_pretrain_corrected_data_splits.csv',
                train_split_name=hyperparams["unlabeled_split_name"],
                val_split_name=hyperparams["val_split_name"],
                test_split_name=hyperparams["test_split_name"],
                train_transforms=train_transforms(teacher_model.input_size, hyperparams["normalize"], cv2.INTER_NEAREST),
                val_transforms=test_val_transforms(teacher_model.input_size, hyperparams["normalize"], cv2.INTER_NEAREST),
                test_transforms=test_val_transforms(teacher_model.input_size, hyperparams["normalize"], cv2.INTER_NEAREST),
                image_path_name='Path',
                target_name='Label',
                split_name='Split',
                batch_size=hyperparams["batch_size_unlabeled"],
                num_workers=hyperparams["workers"],
                shuffle_train=True,
                resampler=identity_resampler,
                pretraining=False,
                binary=False
    )
    unlabeled_datamodule.prepare_data()
    
    finetune_datamodule = EyeDiseaseDataModule(
                csv_path=r'../input/corrected-data-splits/kaggle_pretrain_corrected_data_splits.csv',
                train_split_name=hyperparams["train_split_name"],
                val_split_name=hyperparams["val_split_name"],
                test_split_name=hyperparams["test_split_name"],
                train_transforms=train_transforms(teacher_model.input_size, hyperparams["normalize"], cv2.INTER_NEAREST),
                val_transforms=test_val_transforms(teacher_model.input_size, hyperparams["normalize"], cv2.INTER_NEAREST),
                test_transforms=test_val_transforms(teacher_model.input_size, hyperparams["normalize"], cv2.INTER_NEAREST),
                image_path_name='Path',
                target_name='Label',
                split_name='Split',
                batch_size=hyperparams["batch_size_labeled"],
                num_workers=hyperparams["workers"],
                shuffle_train=True,
                resampler=identity_resampler,
                pretraining=False,
                binary=False
    )
    finetune_datamodule.prepare_data()
    return labeled_datamodule, unlabeled_datamodule, finetune_datamodule


def init_logging():
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
        wandb.init(project=hyperparams["project_name"], entity=hyperparams["entity_name"], config=hyperparams)


def init_distributed():
    if hyperparams["local_rank"] != -1:
        hyperparams["gpu"] = hyperparams["local_rank"]
        torch.distributed.init_process_group(backend='nccl')
        hyperparams["world_size"] = torch.distributed.get_world_size()
    else:
        hyperparams["gpu"] = 0
        hyperparams["world_size"] = 1
    
    if hyperparams["local_rank"] != -1:
        torch.distributed.barrier()
    

def init_models():
    teacher_model = RegNetY3_2gf(4)
    teacher_model = load_lightning_model(teacher_model, hyperparams["lr"], 4)
    student_model = RegNetY3_2gf(4)
    logger.info(f"Teacher Model: {teacher_model.__class__}")
    logger.info(f"Student Model: {student_model.__class__}")
    logger.info(f"Params: {sum(p.numel() for p in teacher_model.parameters())/1e6:.2f}M")
    teacher_model.to(hyperparams["device"])
    student_model.to(hyperparams["device"])
    
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
    return teacher_model, student_model, teacher_optimizer, student_optimizer, teacher_scheduler, student_scheduler, teacher_scaler, student_scaler


def resume(teacher_optimizer, student_optimizer, teacher_scheduler, student_scheduler, teacher_model, 
            student_model, avg_student_model, teacher_scaler, student_scaler):
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


def main():
    init_distributed()
    init_logging()
    labeled_datamodule, unlabeled_datamodule, finetune_datamodule = prepare_data_modules(teacher_model)

    labeled_loader = labeled_datamodule.train_dataloader()
    unlabeled_loader = unlabeled_datamodule.train_dataloader()
    val_loader = labeled_datamodule.val_dataloader()

    if hyperparams["local_rank"] != -1:
        torch.distributed.barrier()    

    avg_student_model = None
    criterion = torch.nn.CrossEntropyLoss(weight=hyperparams["weights"].to(hyperparams["device"]))
    
    teacher_model, student_model, teacher_optimizer, student_optimizer, teacher_scheduler, student_scheduler, teacher_scaler, student_scaler = init_models()

    if hyperparams["resume"]:
        resume(teacher_optimizer, student_optimizer, teacher_scheduler, student_scheduler, 
                teacher_model, student_model, avg_student_model, teacher_scaler, student_scaler)

    if hyperparams['local_rank'] != -1:
        teacher_model = nn.parallel.DistributedDataParallel(
            teacher_model, device_ids=[hyperparams['local_rank']],
            output_device=hyperparams['local_rank'], find_unused_parameters=True)
        student_model = nn.parallel.DistributedDataParallel(
            student_model, device_ids=[hyperparams['local_rank']],
            output_device=hyperparams['local_rank'], find_unused_parameters=True)

    loops = Loops(labeled_loader, unlabeled_loader, val_loader, 
                  finetune_datamodule.train_dataloader(), teacher_model, 
                  student_model, avg_student_model, criterion, teacher_optimizer,
                  student_optimizer, teacher_scheduler, student_scheduler, teacher_scaler, student_scaler)
    
    if hyperparams['finetune']:
        loops.finetune()
        return

    if hyperparams['evaluate']:
        loops.evaluate(student_model, is_student=True)
        return

    teacher_model.zero_grad()
    student_model.zero_grad()
    loops.train_loop()
    return
    

if __name__ == "__main__":
    main()