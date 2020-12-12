# import torchio
# from torchio.transforms import RandomFlip, Compose

import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import random

'''
可以在第一个维度上flip 代表 左右 第二个维度代表前后
'''


class TbiDataSet(Dataset):

    def __init__(self, config, name_list, transform=False):
        self.transform = transform
        self.name_list = name_list
        self.config = config
        self.data_path_list = []
        for name in name_list:
            t1_path_list = glob(f"{config['data_path']}{name}*")
            self.data_path_list.extend(t1_path_list)

    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, idx):
        img_path = self.data_path_list[idx]
        point_path = img_path.replace('norm', 'point')
        aalball_path = img_path.replace('norm', 'aalmap2ball')
        mask_path = img_path.replace('norm', 'label')
        distmap_path = img_path.replace('norm', 'distance_map').replace('nii', 'npy')
        gausmap_path = img_path.replace('norm', 'gaussian_map').replace('nii', 'npy')

        image_sitk = sitk.ReadImage(img_path)
        image = sitk.GetArrayFromImage(image_sitk)

        mask_sitk = sitk.ReadImage(mask_path)
        mask = sitk.GetArrayFromImage(mask_sitk)

        point_sitk = sitk.ReadImage(point_path)
        point = sitk.GetArrayFromImage(point_sitk)

        distmap = np.load(distmap_path)
        ball = distmap <= self.config['dist_thr']

        if self.config['atlas']:
            atlas = sitk.GetArrayFromImage(sitk.ReadImage('DATA/resize/ch2_rpi.nii'))
            atlas_distmap = np.load('DATA/resize/ch2_distance_rpi.npy')
            atlas_ball = atlas_distmap <= self.config['dist_thr']

        # c = image.shape[0]
        # plt.imshow(atlas[c // 2])
        # plt.show()
        # plt.imshow(atlas_ball.sum(0)[c // 2])
        # plt.show()

        if self.transform and random.random() <= 0.5:
            image = image[:, :, ::-1].copy()
            mask = mask[:, :, ::-1].copy()
            point = point[:, :, ::-1].copy()
            ball = ball[:, :, :, ::-1].copy()
            if self.config['atlas']:
                atlas = atlas[:, :, ::-1].copy()
                atlas_ball = atlas_ball[:, :, :, ::-1].copy()

        # plt.imshow(atlas[c // 2])
        # plt.show()
        # plt.imshow(atlas_ball.sum(0)[c // 2])
        # plt.show()

        image = torch.from_numpy(image).unsqueeze(0).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        point = torch.from_numpy(point).unsqueeze(0).float()
        ball = torch.from_numpy(ball).float()


        datadict = {
            'image': image,
            'mask': mask,
            'point': point,
            'ball': ball,
            'path': img_path}

        if self.config['atlas']:
            atlas = torch.from_numpy(atlas).unsqueeze(0).float()
            atlas_ball = torch.from_numpy(atlas_ball).float()
            datadict['atlas'] = atlas
            datadict['atlas_ball'] = atlas_ball
        return datadict


def get_dataloader(config):
    loader_config = config['loaders']
    foldcsv = pd.read_csv(loader_config['csv_path'])
    train_name_list = list(foldcsv[foldcsv['fold'] != loader_config['val_fold']]['name'])
    val_name_list = list(foldcsv[foldcsv['fold'] == loader_config['val_fold']]['name'])
    train_subject_list = []
    val_subject_list = []

    # for name in train_name_list:
    #     t1_path_list = glob(f"{loader_config['data_path']}{name}*")
    #     for t1_path in t1_path_list:
    #         train_subject_list.append(
    #             torchio.Subject(t1=torchio.ScalarImage(t1_path),
    #                             label=torchio.LabelMap(t1_path.replace('norm', 'label'))))
    # for name in val_name_list:
    #     t1_path_list = glob(f"{loader_config['data_path']}{name}*")
    #     for t1_path in t1_path_list:
    #         val_subject_list.append(
    #             torchio.Subject(t1=torchio.ScalarImage(t1_path),
    #                             label=torchio.LabelMap(t1_path.replace('norm', 'label'))))

    # transform = Compose([RandomFlip(axes=('LR', 'AP'))])
    # train_subjects_dataset = torchio.SubjectsDataset(train_subject_list, transform=transform)
    # val_subjects_dataset = torchio.SubjectsDataset(val_subject_list)
    train_subjects_dataset = TbiDataSet(loader_config, train_name_list, transform=False)
    val_subjects_dataset = TbiDataSet(loader_config, val_name_list)
    train_training_loader = DataLoader(train_subjects_dataset, batch_size=loader_config['batch_size'], num_workers=loader_config['num_workers'], shuffle=True)
    val_training_loader = DataLoader(val_subjects_dataset, batch_size=1, num_workers=loader_config['num_workers'], shuffle=False)
    return {'train': train_training_loader, 'val': val_training_loader}


def view_image_with_label(image, label):
    image = (np.array([image, image, image]) * 255).astype(np.int)
    label = np.where(label != 0, 1, 0)
    image = image.transpose(1, 2, 0)
    image[label == 1, 2] = 255
    image[label == 1, 1] = 0
    image[label == 1, 0] = 0
    plt.imshow(image)
    plt.show()

# if __name__ == '__main__':
# train_training_loader, val_training_loader = get_dataloader(1)
# for subjects_batch in train_training_loader:
#     images = subjects_batch['t1'][torchio.DATA]
#     labels = subjects_batch['label'][torchio.DATA]
#     print(subjects_batch['label'][torchio.AFFINE])

# PATH = 'path'
# TYPE = 'type'
# STEM = 'stem'
# DATA = 'data'
# AFFINE = 'affine'
# print(np.unique(labels.numpy()))
# image = images.numpy()[0, 0][:, :, 100]
# label = labels.numpy()[0, 0][:, :, 100]
# view_image_with_label(image, label)
# break
