import os
import importlib
import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from tensorboardX import SummaryWriter

from core.models.model import get_model
from core.loss.losses import get_loss_criterion
from core.metrics import get_evaluation_metric
from core.datasets.TbiDataSet import get_dataloader
from core.utils import get_number_of_learnable_parameters, save_nii

from core.TrainerKeyPoint import TrainerKeyPoint
from . import utils


class TrainerKeyPointMorph(TrainerKeyPoint):

    def __init__(self, config):
        super(TrainerKeyPointMorph, self).__init__(config)

    def initialize(self):

        self.initialize_network()
        self.initialize_optimizer_and_scheduler()
        self.loss_criterion = get_loss_criterion(self.config['loss'])
        self.Morphloss_criterion = [get_loss_criterion(i) for i in self.config['Morphloss']]
        self.eval_criterion = get_evaluation_metric(self.config['eval_metric'])
        self.loaders = get_dataloader(self.config)

        self.num_iterations = 1
        self.num_epoch = 0
        self.best_eval_score = float('-inf') if self.eval_score_higher_is_better else float('+inf')

        if self.amp:
            self.scaler = GradScaler()

    def initialize_network(self):
        self.model = get_model(self.config['model'])
        self.voxelmorph_model = get_model(self.config['Morphmodel'])
        self.logger.info(f'Using {torch.cuda.device_count()} GPUs for training')
        self.model = self.model.cuda()
        self.voxelmorph_model = self.voxelmorph_model.cuda()
        self.logger.info(f'Number of learnable params {get_number_of_learnable_parameters(self.model) + get_number_of_learnable_parameters(self.voxelmorph_model)}')

    def initialize_optimizer_and_scheduler(self):
        assert 'optimizer' in self.config, 'Cannot find optimizer configuration'
        optimizer_config = self.config['optimizer']
        learning_rate = optimizer_config['learning_rate']
        weight_decay = optimizer_config['weight_decay']
        self.optimizer = optim.Adam([{'params': self.model.parameters()},
                                     {'params': self.voxelmorph_model.parameters(), 'lr': learning_rate}],
                                    lr=learning_rate, weight_decay=weight_decay)

        lr_scheduler_config = self.config.get('lr_scheduler', None)
        if lr_scheduler_config is None:
            # use ReduceLROnPlateau as a default scheduler
            return ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=20, verbose=True)
        else:
            lr_scheduler_name = lr_scheduler_config.pop('name')
            lr_scheduler_clazz = getattr(importlib.import_module('torch.optim.lr_scheduler'), lr_scheduler_name)
            # add optimizer to the config
            lr_scheduler_config['optimizer'] = self.optimizer
            self.scheduler = lr_scheduler_clazz(**lr_scheduler_config)

    def train_one_epoch(self, train_loader):
        train_losses = utils.RunningAverage()
        train_eval_distances = utils.RunningAverage()

        self.model.train()

        for i, datadict in enumerate(train_loader):

            image = datadict['image'].cuda()
            label = datadict['ball'].cuda()
            atlas = datadict['atlas'].cuda()
            atlas_label = datadict['atlas_ball'].cuda()

            self.optimizer.zero_grad()
            if self.amp:
                with autocast():
                    output, loss = self._forward_pass(image, label, atlas, atlas_label)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                output, loss = self._forward_pass(image, label, atlas, atlas_label)
                loss.backward()
                self.optimizer.step()

            train_losses.update(loss.item(), self._batch_size(image))

            # log
            if self.num_iterations % self.log_after_iters == 0:
                distance, distance_list = self.eval_criterion(output.cpu().float(), datadict['point'])
                train_eval_distances.update(distance.item(), self._batch_size(image))
                # log stats, params
                self.logger.info(f"Epoch [{str(self.num_epoch).zfill(3)}/{self.max_num_epochs - 1}]. Batch {str(i).zfill(2)}. "
                                 f"TrainIteration {str(self.num_iterations).zfill(4)}. Lr:{self.optimizer.param_groups[0]['lr']:.5f}. "
                                 f"TrainLoss:{train_losses.avg:.5f}. TrainDist:{train_eval_distances.avg:.5f}. ")
                self._log_stats('train', train_losses.avg, train_eval_distances.avg)
                self._log_params()

            # validate
            if self.num_iterations % self.validate_after_iters == 0:
                self.model.eval()
                val_losses, val_distances = self.validate(self.loaders['val'])
                self.model.train()

                eval_score = val_distances.avg
                # adjust learning rate if necessary
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(eval_score)
                else:
                    self.scheduler.step()
                self._log_lr()

                # remember best validation metric
                is_best = self._is_best_eval_score(eval_score)
                self._save_checkpoint(is_best)

                self.logger.info(f'Epoch [{str(self.num_epoch).zfill(3)}/{self.max_num_epochs - 1}]. Batch {str(i).zfill(2)}. TrainIteration {str(self.num_iterations).zfill(4)}. '
                                 f'ValLoss:{val_losses.avg:.5f}. ValDist:{val_distances.avg:.5f}. best {self.best_eval_score:.5f}')

            self.num_iterations += 1

    def _forward_pass(self, image, label, atlas, atlas_label):
        # forward pass
        output = self.model(image)
        output = self.model.keypoint_postprocess(output)
        y_source, pos_flow = self.voxelmorph_model(image, atlas, True)
        y_output = self.voxelmorph_model.transformer(output, pos_flow)

        # compute the loss
        loss_kp = self.loss_criterion(output, label)
        loss_warp_kp = self.loss_criterion(y_output, atlas_label)
        loss_morph = [Morphloss(atlas, y_source) for Morphloss in self.Morphloss_criterion]
        loss = loss_kp + loss_warp_kp + loss_morph[0]
        if len(loss_morph) == 2:
            loss += 0.01 * loss_morph[1]
        return output, loss

    def validate(self, val_loader):
        val_losses = utils.RunningAverage()
        val_distances = utils.RunningAverage()

        with torch.no_grad():
            for i, datadict in enumerate(val_loader):
                image = datadict['image'].cuda()
                label = datadict['ball'].cuda()
                atlas = datadict['atlas'].cuda()
                atlas_label = datadict['atlas_ball'].cuda()

                output, loss = self._forward_pass(image, label, atlas, atlas_label)
                val_losses.update(loss.item(), self._batch_size(image))

                distance, distance_list = self.eval_criterion(output.cpu(), datadict['point'])
                val_distances.update(distance.item(), self._batch_size(image))

            self._log_stats('val', val_losses.avg, val_distances.avg)

            return val_losses, val_distances
