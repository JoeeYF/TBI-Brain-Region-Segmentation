import os
import importlib
from SimpleITK.SimpleITK import LandmarkBasedTransformInitializer
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

from . import utils


class TrainerKeyPoint:

    def __init__(self, config):

        self.config = config
        self.save_dir = config['base_dir']
        trainer_config = config['trainer']
        self.max_num_epochs = trainer_config['epochs']
        self.validate_after_iters = trainer_config['validate_after_iters']
        self.log_after_iters = trainer_config['log_after_iters']
        self.eval_score_higher_is_better = trainer_config['eval_score_higher_is_better']
        self.amp = trainer_config['amp']

        self.logger = utils.get_logger('UNet3DTrainer', self.save_dir + '/file.log')

        for key, value in config.items():
            self.logger.info(f'{key}: {value}')

        self.initialize()

    def initialize(self):

        self.initialize_network()
        self.initialize_optimizer_and_scheduler()
        self.loss_criterion = get_loss_criterion(self.config['loss'])
        self.eval_criterion = get_evaluation_metric(self.config['eval_metric'])
        self.loaders = get_dataloader(self.config)

        self.num_iterations = 1
        self.num_epoch = 0
        self.best_eval_score = float('-inf') if self.eval_score_higher_is_better else float('+inf')

        if self.amp:
            self.scaler = GradScaler()

    def initialize_network(self):
        self.model = get_model(self.config['model'])
        self.logger.info(f'Using {torch.cuda.device_count()} GPUs for training')
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
        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, 'logs'))
        self.nobest_times = 0
        for _ in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            self.train_one_epoch(self.loaders['train'])
            self.num_epoch += 1
            if self.nobest_times > 10:
                self.logger.info(f"nobest_times > 10. Stop training...")
                return
        self.logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")

    def train_one_epoch(self, train_loader):
        train_losses = utils.RunningAverage()
        train_eval_distances = utils.RunningAverage()
        self.model.train()

        for i, datadict in enumerate(train_loader):

            image = datadict['image'].cuda()
            label = datadict['ball'].cuda()

            self.optimizer.zero_grad()
            if self.amp:
                with autocast():
                    output, loss = self._forward_pass(image, label)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                output, loss = self._forward_pass(image, label)
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

    def _forward_pass(self, image, label, **kwargs):
        # forward pass
        output = self.model(image)
        output = self.model.keypoint_postprocess(output)
        # compute the loss
        loss = self.loss_criterion(output, label)

        return output, loss

    def validate(self, val_loader):
        # self.logger.info('Validating...')

        val_losses = utils.RunningAverage()
        val_distances = utils.RunningAverage()

        with torch.no_grad():
            for i, datadict in enumerate(val_loader):
                image = datadict['image'].cuda()
                label = datadict['ball'].cuda()

                output, loss = self._forward_pass(image, label)
                val_losses.update(loss.item(), self._batch_size(image))

                distance, distance_list = self.eval_criterion(output.cpu(), datadict['point'])
                val_distances.update(distance.item(), self._batch_size(image))

            self._log_stats('val', val_losses.avg, val_distances.avg)

            return val_losses, val_distances

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            self.best_eval_score = eval_score
            self.nobest_times = 0
        else:
            self.nobest_times += 1

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
        self.infer_model.cuda()
        self.infer_model.eval()

        state = torch.load(os.path.join(self.save_dir, 'best_checkpoint.pytorch'), map_location='cpu')
        self.logger.info(f"Load {os.path.join(self.save_dir, 'best_checkpoint.pytorch')}...")
        self.logger.info(f"Epoch:{state['epoch']} Iter:{state['num_iterations']} Score:{state['best_eval_score']}")
        self.infer_model.load_state_dict(state['model_state_dict'])

        self.output_dir = os.path.join(self.save_dir, 'prediction')
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"Running prediction on {len(self.loaders['val'])} batches...")

        distanceList = []
        distancesList = []
        nameList = []

        with torch.no_grad():
            for _, datadict in enumerate(self.loaders['val']):
                image = datadict['image'].cuda()
                img_path = datadict['path'][0]
                name = img_path.split('/')[-1].split('.')[0]

                # forward pass
                output = self.infer_model(image)
                output = self.infer_model.keypoint_postprocess(output)
                distance, distance_list = self.eval_criterion(output.cpu(), datadict['point'])
                image = image.cpu().numpy()[0][0]
                output = output.cpu().numpy()[0]

                self.logger.info(f'{name}, {distance}')
                distanceList.append(distance.item())
                distancesList.append(distance_list)
                nameList.append(name)

                pred_ballmap = np.zeros_like(image)
                label_ballmap = np.zeros_like(image)

                for classindex in range(17):
                    output_c = output[classindex]
                    output_c_coord = np.where(output_c == np.max(output_c))
                    a = int(output_c_coord[0].mean())
                    b = int(output_c_coord[1].mean())
                    c = int(output_c_coord[2].mean())
                    coord = np.array(np.where(image.squeeze() < 1e10))
                    coord = np.stack(coord, axis=1).reshape(image.shape[0], image.shape[1], image.shape[2], 3)
                    pred_distmap = np.sqrt(((coord - np.array([a, b, c]).squeeze()) ** 2).sum(-1))
                    pred_ballmap[pred_distmap < 4] = classindex + 1
                    label_ballmap = label_ballmap + datadict['ball'].numpy()[0][classindex] * (classindex + 1)

                save_nii(image, img_path, f'{self.output_dir}/{name}_image.nii')
                save_nii(label_ballmap, img_path, f'{self.output_dir}/{name}_label_ball.nii')
                save_nii(pred_ballmap, img_path, f'{self.output_dir}/{name}_pred_ball.nii')

            distancearray = np.array(distancesList)
            self.logger.info(f'mean/std: {np.mean(distanceList):.5f}, {np.std(distanceList):.5f}')
            distancedict = {f'{i + 1}': distancearray[:, i] for i in range(17)}

            df = pd.DataFrame({'distance': distanceList, **distancedict}, index=nameList)
            df.index.name = 'name'
            df.to_csv(self.save_dir + '/keypoint.csv')
