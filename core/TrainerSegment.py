import os
import importlib
import numpy as np
import pandas as pd 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tensorboardX import SummaryWriter

from core.models.model import get_model
from core.loss.losses import get_loss_criterion
from core.metrics import get_evaluation_metric
from core.datasets.TbiDataSet import get_dataloader
from core.utils import get_number_of_learnable_parameters, save_nii,cal_dice

from . import utils


class TrainerSegment:

    def __init__(self, config):

        self.config = config
        self.save_dir = config['base_dir']

        trainer_config = config['trainer']
        self.max_num_epochs = trainer_config['epochs']
        self.validate_after_iters = trainer_config['validate_after_iters']
        self.log_after_iters = trainer_config['log_after_iters']
        self.eval_score_higher_is_better = trainer_config['eval_score_higher_is_better']

        self.logger = utils.get_logger('UNet3DTrainer', self.save_dir+'/file.log')
        for key, value in config.items():
            self.logger.info(f'{key}: {value}')
        self.initialize()

    def initialize(self):

        self.initialize_network()
        self.initialize_optimizer_and_scheduler()
        self.loss_criterion = get_loss_criterion(self.config['loss'])
        self.eval_criterion = get_evaluation_metric(self.config['eval_metric'])
        self.loaders = get_dataloader(self.config)

        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, 'logs'))
        self.num_iterations = 1
        self.num_epoch = 0
        self.best_eval_score = float('-inf') if self.eval_score_higher_is_better else float('+inf')

    def initialize_network(self):
        self.model = get_model(self.config['model'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f'Using {torch.cuda.device_count()} GPUs for training')
        if torch.cuda.device_count() > 1 and not self.device.type == 'cpu':
            self.model = nn.DataParallel(self.model)
        self.logger.info(f"Sending the model to '{self.device}'")
        self.model = self.model.cuda()
        self.logger.info(f'Number of learnable params {get_number_of_learnable_parameters(self.model)}')

    def initialize_optimizer_and_scheduler(self):
        assert 'optimizer' in self.config, 'Cannot find optimizer configuration'
        optimizer_config = self.config['optimizer']
        learning_rate = optimizer_config['learning_rate']
        weight_decay = optimizer_config['weight_decay']
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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

    def fit(self):
        for _ in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            self.train_one_epoch(self.loaders['train'])
            self.num_epoch += 1
        self.logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")

    def train_one_epoch(self, train_loader):
        train_losses = utils.RunningAverage()
        train_eval_scores = utils.RunningAverage()
        self.model.train()

        for i, (image, mask, point, distmap, img_path) in enumerate(train_loader):

            image = image.cuda()
            label = mask.cuda()
            label = utils.expand_as_one_hot(label, 18)
            output, loss = self._forward_pass(image, label)
            train_losses.update(loss.item(), self._batch_size(image))

            # compute gradients and update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log
            if self.num_iterations % self.log_after_iters == 0:
                output = self.model.final_activation(output)
                pred = torch.argmax(output, dim=1)
                pred_one_hot = utils.expand_as_one_hot(pred, 18)
                eval_score = self.eval_criterion(pred_one_hot, label)
                train_eval_scores.update(eval_score.item(), self._batch_size(image))

                # log stats, params
                self.logger.info(f'Epoch [{str(self.num_epoch).zfill(3)}/{self.max_num_epochs - 1}]. Batch {str(i).zfill(2)}. TrainIteration {str(self.num_iterations).zfill(4)}. '
                                 f"Lr:{self.optimizer.param_groups[0]['lr']:.5f}. TrainLoss:{train_losses.avg:.5f}. EvaluationScore:{train_eval_scores.avg:.5f}")
                self._log_stats('train', train_losses.avg, train_eval_scores.avg)
                self._log_params()

            # validate
            if self.num_iterations % self.validate_after_iters == 0:
                self.model.eval()
                val_losses, val_scores, val_distances = self.validate(self.loaders['val'])
                self.model.train()

                eval_score = val_scores.avg
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
                                 f'ValLoss:{val_losses.avg:.5f}. ValScore:{val_scores.avg:.5f}. best {self.best_eval_score:.5f}')

            self.num_iterations += 1

    def _forward_pass(self, image, label):
        # forward pass
        output = self.model(image)
        # compute the loss
        loss = self.loss_criterion(output, label)

        return output, loss

    def validate(self, val_loader):
        # self.logger.info('Validating...')

        val_losses = utils.RunningAverage()
        val_scores = utils.RunningAverage()
        val_distances = utils.RunningAverage()

        with torch.no_grad():
            for i, (image, mask, point, distmap, img_path) in enumerate(val_loader):

                image = image.cuda()
                label = mask.cuda()
                label = utils.expand_as_one_hot(label, 18)
                output, loss = self._forward_pass(image, label, None)
                val_losses.update(loss.item(), self._batch_size(image))

                output = self.model.final_activation(output)
                pred = torch.argmax(output, dim=1)
                pred_one_hot = utils.expand_as_one_hot(pred, 18)
                eval_score = self.eval_criterion(pred_one_hot, label)
                val_scores.update(eval_score.item(), self._batch_size(image))

            self._log_stats('val', val_losses.avg, val_scores.avg)

            return val_losses, val_scores, val_distances

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            self.best_eval_score = eval_score

        return is_best

    def _save_checkpoint(self, is_best):

        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        utils.save_checkpoint(state={'epoch': self.num_epoch + 1,
                                     'num_iterations': self.num_iterations,
                                     'model_state_dict': state_dict,
                                     'best_eval_score': self.best_eval_score,
                                     'eval_score_higher_is_better': self.eval_score_higher_is_better,
                                     'optimizer_state_dict': self.optimizer.state_dict(),
                                     'device': str(self.device),
                                     'max_num_epochs': self.max_num_epochs,
                                     'validate_after_iters': self.validate_after_iters,
                                     'log_after_iters': self.log_after_iters},
                              is_best=is_best,
                              checkpoint_dir=self.save_dir,
                              logger=self.logger)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_stats(self, phase, loss_avg, eval_score_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_eval_score_avg': eval_score_avg
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    @staticmethod
    def _batch_size(image):
        if isinstance(image, list) or isinstance(image, tuple):
            return image[0].size(0)
        else:
            return image.size(0)

    def predict(self):
        self.infer_model = get_model(self.config['model'])
        self.infer_model.eval()
        self.infer_model.cuda()
        self.infer_model.testing = True
        state = torch.load(os.path.join(self.save_dir, 'best_checkpoint.pytorch'), map_location='cpu')
        self.infer_model.load_state_dict(state['model_state_dict'])
        self.output_dir = os.path.join(self.save_dir, 'prediction')
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"Running prediction on {len(self.loaders['val'])} batches...")

        with torch.no_grad():
            for i, (image, mask, point, distmap, img_path) in enumerate(self.loaders['val']):
                path = img_path[0]
                name = path.split('/')[-1].split('.')[0]
                image = image.to(self.device)
                label = mask.to(self.device)
                self.name_list.append(name)

                # forward pass
                output = self.model(image)
                pred = torch.argmax(output, dim=1)
                image = image.cpu().numpy()[0][0]
                label = label.cpu().numpy()[0][0]
                pred = pred.cpu().float().numpy()[0]

                dice, dice_per_channel = cal_dice(pred, label)
                self.dice_list.append(dice)
                self.dice_per_channel_list.append(dice_per_channel)
                
                self.logger.info(f'{name}, {dice}')
                save_nii(image, path, f'{self.output_dir}/{name}_image.nii')
                save_nii(label, path, f'{self.output_dir}/{name}_label_mask.nii')
                save_nii(pred, path, f'{self.output_dir}/{name}_pred_mask.nii')
                
            self.logger.info(f'{np.mean(self.dice_list)}')

            df = pd.DataFrame(np.array(self.dice_per_channel_list),columns=[str(i) for i in range(17)],index=self.name_list)
            df.to_csv(self.base_dir+'/dice.csv')

