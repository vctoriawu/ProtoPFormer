# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
from cProfile import label
import math
import os
import sys
import logging
import pickle
from typing import Iterable, Optional
import wandb

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from torch.nn.modules.loss import _Loss
import torch.nn as nn
import tools.utils as utils
from torch.utils.tensorboard import SummaryWriter


def compute_cluster(min_distances, target, num_classes, loss_weight=0.8, reduction='mean'):
    if loss_weight == 0:
        return torch.tensor(0, device=target.device)

    target_one_hot = nn.functional.one_hot(target, num_classes=num_classes)  # shape (N, classes)
    min_distances = min_distances.reshape((min_distances.shape[0], num_classes, -1))
    class_specific_min_distances, _ = min_distances.min(dim=2)  # Shape = (N, classes)
    positives = class_specific_min_distances * target_one_hot  # shape (N, classes)

    if reduction == "mean":
        loss = positives.sum(dim=1).mean()
    elif reduction == "sum":
        loss = positives.sum()

    return loss_weight * loss

def compute_separation(min_distances, target, num_classes, strategy='all', normalize=True, loss_weight=-0.08, reduction='mean'):
    if loss_weight == 0:
        return torch.tensor(0, device=target.device)

    target_one_hot = nn.functional.one_hot(target, num_classes=num_classes)
    min_distances = min_distances.reshape((min_distances.shape[0], num_classes, -1))
    class_specific_min_distances, _ = min_distances.min(dim=2)  # Shape = (N, classes)
    negatives = class_specific_min_distances * (1 - target_one_hot)  # shape (N, classes)
    normalized_negatives = negatives/torch.sum((1 - target_one_hot), dim=1, keepdim=True)

    if strategy == "closest":
        # this is based on protopnet paper and results the same, but their code implementation is different
        # penalize against the closest class of "other" clusters, similar to ProtoPNet's loss equation in paper
        negatives[negatives == 0] = torch.tensor(float('inf')).to(negatives.device)
        negatives, _ = negatives.min(dim=1)  # shape = (N)
    else:
        if normalize:
            negatives = normalized_negatives

    if reduction == "mean":
        loss = negatives.sum(dim=1).mean()
    elif reduction == "sum":
        loss = negatives.sum()

    return loss_weight * loss  # loss weight has to be negative here

def train_one_epoch(model: torch.nn.Module, criterion: _Loss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    tb_writer: SummaryWriter, iteration: int,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    ent_loss = None,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    args=None,
                    set_training_mode=True,):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 30
    num_classes = 200

    logger = logging.getLogger("train")
    logger.info("Start train one epoch")
    it = 0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs, auxi_item, distances = model(samples)

            loss = criterion(outputs, targets)
            entailment_loss = ent_loss.compute(model.module)

            if args.use_ppc_loss:
                ppc_cov_coe, ppc_mean_coe = args.ppc_cov_coe, args.ppc_mean_coe
                total_proto_act, cls_attn_rollout, original_fea_len = auxi_item[2], auxi_item[3], auxi_item[4]
                if hasattr(model, 'module'):
                    ppc_cov_loss, ppc_mean_loss = model.module.get_PPC_loss(total_proto_act, cls_attn_rollout, original_fea_len, targets)
                else:
                    ppc_cov_loss, ppc_mean_loss = model.get_PPC_loss(total_proto_act, cls_attn_rollout, original_fea_len, targets)

                ppc_cov_loss = ppc_cov_coe * ppc_cov_loss
                ppc_mean_loss = ppc_mean_coe * ppc_mean_loss
                #if epoch >= 20:
                #    loss = loss + ppc_cov_loss + ppc_mean_loss
            #############################################
            #### Our own cluster and separation loss ####
            #############################################
            ## My version of Cluster and Separation losses ###
            # TODO check to see if mine is causing the NAN error!
            # cluster cost
            cluster_cost = compute_cluster(distances[1], targets, num_classes=num_classes)
            # separation cost
            separation_cost = compute_separation(distances[1], targets, num_classes=num_classes)

            # Global Cluster and Separation losses
            # TODO check if loss computation is compatible with these global features
            # cluster cost
            cluster_cost_global = compute_cluster(distances[0], targets, num_classes=num_classes)
            # separation cost
            separation_cost_global = compute_separation(distances[0], targets, num_classes=num_classes)

        #if epoch >= 20:
        #    loss = loss + 0.1*entailment_loss + 0.1*0.5*(cluster_cost + cluster_cost_global + separation_cost + separation_cost_global)
        #if epoch >= 20:
        loss = loss + entailment_loss + (1e-3)*(cluster_cost + cluster_cost_global + separation_cost + separation_cost_global)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        loss_scaler(loss, optimizer, clip_grad=max_norm,
                   parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        # Print gradients for the last layer
        #if model.module.last_layer.weight.grad is not None:
        #    print("Gradient for last layer weights:", model.module.last_layer.weight.grad)
        #if model.module.last_layer_global.weight.grad is not None:
        #    print("Gradient for last layer weights:", model.module.last_layer_global.weight.grad)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        tb_writer.add_scalars(
            main_tag="train/loss",
            tag_scalar_dict={
                "cls": loss.item(),
            },
            global_step=iteration+it
        )
        if args.use_global and args.use_ppc_loss:
            tb_writer.add_scalars(
                main_tag="train/ppc_cov_loss",
                tag_scalar_dict={
                    "ppc_cov_loss": ppc_cov_loss.item(),
                },
                global_step=iteration+it
            )
            tb_writer.add_scalars(
                main_tag="train/ppc_mean_loss",
                tag_scalar_dict={
                    "ppc_mean_loss": ppc_mean_loss.item(),
                },
                global_step=iteration+it
            )
        it += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    # Log each averaged metric to WandB
    wandb.log({f'train/{k}': meter.global_avg for k, meter in metric_logger.meters.items()})
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def get_img_mask(data_loader, model, device, args):
    logger = logging.getLogger("get mask")
    logger.info("Get mask")
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Get Mask:'

    # switch to evaluation mode
    model.eval()

    all_attn_mask = []
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            cat_mask = model.get_attn_mask(images)
            all_attn_mask.append(cat_mask.cpu())
    all_attn_mask = torch.cat(all_attn_mask, dim=0) # (num, 2, 14, 14)
    if hasattr(model, 'module'):
        model.module.all_attn_mask = all_attn_mask
    else:
        model.all_attn_mask = all_attn_mask

@torch.no_grad()
def evaluate(data_loader, model, device, args):
    logger = logging.getLogger("validate")
    logger.info("Start validation")
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    all_token_attn, pred_labels = [], []
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output, auxi_items = model(images,)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        _, pred = output.topk(k=1, dim=1)
        pred_labels.append(pred)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        if args.use_global:
            global_acc1 = accuracy(auxi_items[2], target)[0]
            local_acc1 = accuracy(auxi_items[3], target)[0]
            metric_logger.meters['global_acc1'].update(global_acc1.item(), n=batch_size)
            metric_logger.meters['local_acc1'].update(local_acc1.item(), n=batch_size)
        all_token_attn.append(auxi_items[0])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    # Log each averaged metric to WandB
    wandb.log({f'test/{k}': meter.global_avg for k, meter in metric_logger.meters.items()})
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}