import os
import importlib
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
from core.utils import get_number_of_learnable_parameters, save_nii, cal_dice

from . import utils


class Dice:
    """
    N-D dice for segmentation
    """

    def __call__(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return dice


class TrainerSegmentMorph:

    def __init__(self, config):

        self.config = config
        self.save_dir = config['base_dir']
        trainer_config = config['trainer']
        self.max_num_epochs = trainer_config['epochs']
        self.validate_after_iters = trainer_config['validate_after_iters']
        self.log_after_iters = trainer_config['log_after_iters']
        self.eval_score_higher_is_better = trainer_config['eval_score_higher_is_better']
        self.stop_after_nobest_iters = trainer_config['stop_after_nobest_iters']
        self.amp = trainer_config['amp']

        self.logger = utils.get_logger('UNet3DTrainer', self.save_dir + '/file.log')

        for key, value in config.items():
            self.logger.info(f'{key}: {value}')

        self.initialize()

    def initialize(self):

        self.initialize_network()
        self.initialize_optimizer_and_scheduler()
        self.loss_criterion = get_loss_criterion(self.config['loss'])
        self.kploss_criterion = get_loss_criterion({'name': 'KeyPointBCELoss'})
        self.Morphloss_criterion = [get_loss_criterion(i) for i in self.config['Morphloss']]
        self.eval_criterion = get_evaluation_metric(self.config['eval_metric'])
        self.loaders = get_dataloader(self.config)

        self.num_iterations = 1
        self.num_epoch = 0
        self.best_eval_score = float('-inf') if self.eval_score_higher_is_better else float('+inf')

        if self.amp:
            self.scaler = GradScaler()

    def initialize_network(self):
        # self.model = get_model(self.config['model'])
        self.voxelmorph_model = get_model(self.config['Morphmodel'])
        self.logger.info(f'Using {torch.cuda.device_count()} GPUs for training')
        # self.model = self.model.cuda()
        self.voxelmorph_model = self.voxelmorph_model.cuda()
        self.logger.info(f'Number of learnable params {get_number_of_learnable_parameters(self.voxelmorph_model)}')

    def initialize_optimizer_and_scheduler(self):
        assert 'optimizer' in self.config, 'Cannot find optimizer configuration'
        optimizer_config = self.config['optimizer']
        learning_rate = optimizer_config['learning_rate']
        weight_decay = optimizer_config['weight_decay']
        self.optimizer = optim.Adam(self.voxelmorph_model.parameters(),
                                    lr=learning_rate*10, weight_decay=weight_decay)
        lr_scheduler_config = self.config.get('lr_scheduler', None)
        lr_scheduler_name = lr_scheduler_config.pop('name')
        lr_scheduler_clazz = getattr(importlib.import_module('torch.optim.lr_scheduler'), lr_scheduler_name)
        self.scheduler = lr_scheduler_clazz(self.optimizer, **lr_scheduler_config)

    def fit(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, 'logs'))
        self.nobest_times = 0

        for _ in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            self.train_one_epoch(self.loaders['train'])
            self.num_epoch += 1
            if self.nobest_times > self.stop_after_nobest_iters:
                self.logger.info(f"nobest_times > {self.stop_after_nobest_iters}. Stop training...")
                return

        self.logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")

    def train_one_epoch(self, train_loader):
        train_losses = utils.RunningAverage()
        train_eval_scores = utils.RunningAverage()

        train_morph_dices = utils.RunningAverage()

        train_seg_losses = utils.RunningAverage()
        train_ncc_losses = utils.RunningAverage()
        train_grad_losses = utils.RunningAverage()
        train_warpseg_losses = utils.RunningAverage()
        train_kp_losses = utils.RunningAverage()

        self.voxelmorph_model.train()

        for i, datadict in enumerate(train_loader):

            image = datadict['image'].cuda()
            label = utils.expand_as_one_hot(datadict['mask'], 18).cuda()
            kplabels = [datadict['ball'].cuda(), datadict['ball2'].cuda()]

            atlas = datadict['atlas'].cuda()
            atlas_label = utils.expand_as_one_hot(datadict['atlas_mask'], 18).cuda()

            self.optimizer.zero_grad()
            if self.amp:
                with autocast():
                    output, loss, loss_dict = self._forward_pass(image, label, kplabels, atlas, atlas_label)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                output, loss, loss_dict = self._forward_pass(image, label, kplabels, atlas, atlas_label)
                loss.backward()
                self.optimizer.step()

            train_losses.update(loss.item(), self._batch_size(image))
            train_seg_losses.update(loss_dict['SegLoss'], self._batch_size(image))
            train_kp_losses.update(loss_dict['KpLoss'], self._batch_size(image))
            train_ncc_losses.update(loss_dict['NCCLoss'], self._batch_size(image))
            train_grad_losses.update(loss_dict['GradLoss'], self._batch_size(image))
            train_warpseg_losses.update(loss_dict['WarpSegLoss'], self._batch_size(image))
            train_morph_dices.update(loss_dict['morphDice'], self._batch_size(image))

            train_score = self.eval_criterion(output.float(), label)
            train_eval_scores.update(train_score.item(), self._batch_size(image))

            # log
            if self.num_iterations % self.log_after_iters == 0:
                # log stats, params
                self.logger.info(f"Epoch [{str(self.num_epoch).zfill(3)}/{self.max_num_epochs - 1}]. Batch {str(i).zfill(2)}. "
                                 f"TrainIteration {str(self.num_iterations).zfill(4)}. Lr:{self.optimizer.param_groups[0]['lr']:.5f}. "
                                 f"SegLoss {train_seg_losses.avg:.5f}. KpLoss {train_kp_losses.avg:.5f}. MorphDice:{train_morph_dices.avg:.5f}. "
                                 f"NCCLoss:{train_ncc_losses.avg:.5f}. GradLoss {train_grad_losses.avg:.5f}. WarpSegLoss:{train_warpseg_losses.avg:.5f}. "
                                 f"TrainLoss:{train_losses.avg:.5f}. TrainScore:{train_eval_scores.avg:.5f}. ")

                self._log_stats('train', train_losses.avg, train_eval_scores.avg)
                # self._log_params()

            # validate
            if self.num_iterations % self.validate_after_iters == 0:
                self.voxelmorph_model.eval()
                val_losses, val_scores = self.validate(self.loaders['val'])
                self.voxelmorph_model.train()

                eval_score = val_scores.avg
                # adjust learning rate if necessary
                self._log_lr()
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(eval_score)
                else:
                    self.scheduler.step()

                # remember best validation metric
                is_best = self._is_best_eval_score(eval_score)
                self._save_checkpoint(is_best)

                self.logger.info(f'Epoch [{str(self.num_epoch).zfill(3)}/{self.max_num_epochs - 1}]. Batch {str(i).zfill(2)}. TrainIteration {str(self.num_iterations).zfill(4)}. '
                                 f'ValLoss:{val_losses.avg:.5f}. ValScore:{val_scores.avg:.5f}. best {self.best_eval_score:.5f}')

            self.num_iterations += 1

    def _forward_pass(self, image, label, kplabel, atlas, atlas_label, **kwargs):
        # encoders_heatmaps, decoders_heatmaps, output = self.model(image)
        y_source, pos_flow = self.voxelmorph_model(image, atlas, True)
        # y_output = self.voxelmorph_model.transformer(output, pos_flow)
        y_label = self.voxelmorph_model.transformer(label, pos_flow)
        output = self.voxelmorph_model.transformer(atlas_label, -pos_flow)
        # # seg loss for unet
        # lossseg = self.loss_criterion(output, label)
        #
        # # kp loss for mpb
        # losskp = self.kploss_criterion(encoders_heatmaps[0], kplabel[0])
        # losskp += self.kploss_criterion(decoders_heatmaps[-1], kplabel[0])
        # if len(decoders_heatmaps) == 2:
        #     losskp += self.kploss_criterion(encoders_heatmaps[1], kplabel[1])
        #     losskp += self.kploss_criterion(decoders_heatmaps[-2], kplabel[1])
        # lossskp = losskp / len(decoders_heatmaps) / 2

        # morph loss for voxelmorph
        loss_morphs = [Morphloss(atlas, y_source) for Morphloss in self.Morphloss_criterion]
        loss_morph_mask = nn.MSELoss()(y_label, atlas_label)
        loss_morph = loss_morphs[0] + 0.01 * loss_morphs[1] + loss_morph_mask

        # loss for warp_seg
        # loss_warp_seg = self.loss_criterion(y_output.softmax(dim=1), atlas_label)

        morphDice = Dice()(y_source, atlas)


        loss = loss_morph

        loss_dict = {
            # 'SegLoss': lossseg.item(),
            #          'KpLoss': lossskp.item(),
                     'NCCLoss': loss_morphs[0].item(),
                     'GradLoss': loss_morphs[1].item(),
                     # 'WarpSegLoss': loss_warp_seg.item(),
                     'morphDice': morphDice.item()
                     }
        return output, loss, loss_dict

    def validate(self, val_loader):
        # self.logger.info('Validating...')

        val_losses = utils.RunningAverage()
        val_scores = utils.RunningAverage()

        with torch.no_grad():
            for i, datadict in enumerate(val_loader):
                image = datadict['image'].cuda()
                label = utils.expand_as_one_hot(datadict['mask'], 18).cuda()
                kplabels = [datadict['ball'].cuda(), datadict['ball2'].cuda()]

                atlas = datadict['atlas'].cuda()
                atlas_label = utils.expand_as_one_hot(datadict['atlas_mask'], 18).cuda()

                output, loss, _ = self._forward_pass(image, label, kplabels, atlas, atlas_label)

                val_losses.update(loss.item(), self._batch_size(image))

                eval_score = self.eval_criterion(output, label)
                val_scores.update(eval_score.item(), self._batch_size(image))

            self._log_stats('val', val_losses.avg, val_scores.avg)

            return val_losses, val_scores

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
                                     'log_after_iters': self.log_after_iters
                                     },
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
        for name, value in self.voxelmorph_model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    @staticmethod
    def _batch_size(image):
        if isinstance(image, list) or isinstance(image, tuple):
            return image[0].size(0)
        else:
            return image.size(0)

    def predict(self):
        self.infer_model = get_model(self.config['Morphmodel'])
        self.infer_model.eval()
        self.infer_model.cuda()
        self.infer_model.testing = True
        state = torch.load(os.path.join(self.save_dir, 'best_checkpoint.pytorch'), map_location='cpu')
        self.infer_model.load_state_dict(state['model_state_dict'])
        self.output_dir = os.path.join(self.save_dir, 'prediction')
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"Running prediction on {len(self.loaders['val'])} batches...")
        dice_list = []
        dice_per_channel_list = []
        name_list = []
        with torch.no_grad():
            for _, datadict in enumerate(self.loaders['val']):
                img_path = datadict['path'][0]
                name = img_path.split('/')[-1].split('.')[0]
                name_list.append(name)

                image = datadict['image'].cuda()
                label = utils.expand_as_one_hot(datadict['mask'], 18).cuda()
                kplabels = [datadict['ball'].cuda(), datadict['ball2'].cuda()]

                atlas = datadict['atlas'].cuda()
                atlas_label = utils.expand_as_one_hot(datadict['atlas_mask'], 18).cuda()

                output, loss, _ = self._forward_pass(image, label, kplabels, atlas, atlas_label)

                # output = self.infer_model.segment_postprocess(output)
                pred = torch.argmax(output, dim=1)
                image = image.cpu().numpy()[0][0]
                pred = pred.cpu().numpy()[0]
                label = label.numpy()[0][0]

                dice, dice_per_channel = cal_dice(pred, label)
                dice_list.append(dice)
                dice_per_channel_list.append(dice_per_channel)
                self.logger.info(f'{name}, {dice}')
                save_nii(image, img_path, f'{self.output_dir}/{name}_image.nii')
                save_nii(label, img_path, f'{self.output_dir}/{name}_label_mask.nii')
                # for i in range(1, 18):
                save_nii(pred, img_path, f'{self.output_dir}/{name}_pred_mask.nii')

                # break

            self.logger.info(f'{np.mean(dice_list)}')

            dice_per_channel_array = np.array(dice_per_channel_list)
            dice_per_channel_dict = {f'{i + 1}': dice_per_channel_array[:, i] for i in range(17)}
            df = pd.DataFrame({'dice': dice_list, **dice_per_channel_dict}, index=name_list)
            df.index.name = 'name'
            df.to_csv(self.save_dir + '/dice.csv')
