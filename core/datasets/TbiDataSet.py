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


class SegmentDataSet(Dataset):

    def __init__(self, config, name_list, transform=False):
        self.is_segmentation = config['is_segmentation']
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
        mask_path = img_path.replace('norm', 'label')
        img_name = img_path.split('/')[-1].split('.')[0]
        distmap_path = img_path.replace('norm', 'distance_map').replace('nii', 'npy')
        distmap2_path = distmap_path.replace('resize', 'resize2')
        image_sitk = sitk.ReadImage(img_path)
        image = sitk.GetArrayFromImage(image_sitk)

        mask_sitk = sitk.ReadImage(mask_path)
        mask = sitk.GetArrayFromImage(mask_sitk)

        image = torch.from_numpy(image).unsqueeze(0).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        # keypoint_path = glob(f'DATA/resize/ball3/{img_name}.nii')
        # assert len(keypoint_path) == 1
        # kp_sitk = sitk.ReadImage(keypoint_path[0])
        # kp = sitk.GetArrayFromImage(kp_sitk)
        # kp = torch.from_numpy(kp).unsqueeze(0).float()
        # image = torch.cat([image, kp], dim=0)

        datadict = {
            'image': image,
            'mask': mask,
            'path': img_path,
        }

        if self.config['use_keypoint']:
            distmap = np.load(distmap_path)
            ball = distmap <= self.config['dist_thr']
            ball = torch.from_numpy(ball).float()

            distmap2 = np.load(distmap2_path)
            ball2 = distmap2 <= self.config['dist_thr'] - 1
            ball2 = torch.from_numpy(ball2).float()

            datadict['ball'] = ball
            datadict['ball2'] = ball2

        if self.config['atlas']:
            atlas = sitk.GetArrayFromImage(sitk.ReadImage(img_path.replace('norm', 'antsnorm')))
            atlas_mask = sitk.GetArrayFromImage(sitk.ReadImage(img_path.replace('label', 'antslabel'))).astype(np.float)
            atlas = torch.from_numpy(atlas).unsqueeze(0).float()
            atlas_mask = torch.from_numpy(atlas_mask).unsqueeze(0).float()
            datadict['atlas'] = atlas
            datadict['atlas_mask'] = atlas_mask

        return datadict


class KeyPointDataSet(Dataset):

    def __init__(self, config, name_list, transform=False):
        self.transform = transform
        self.name_list = name_list
        self.config = config
        self.data_path_list = []
        self.config['gaussian'] = self.config.get('gaussian', False)
        for name in name_list:
            t1_path_list = glob(f"{config['data_path']}{name}*")
            self.data_path_list.extend(t1_path_list)

    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, idx):
        img_path = self.data_path_list[idx]
        mask_path = img_path.replace('norm', 'label')
        point_path = img_path.replace('norm', 'point')
        distmap_path = img_path.replace('norm', 'distance_map').replace('nii', 'npy')
        gausmap_path = img_path.replace('norm', 'gaussian_map2').replace('nii', 'npy')
        cluster_gausmap_path = img_path.replace('norm', 'cluster_gaussian_map').replace('nii', 'npy')
        cluster_ball_path = img_path.replace('norm', 'cluster_ball')

        image_sitk = sitk.ReadImage(img_path)
        image = sitk.GetArrayFromImage(image_sitk)

        mask_sitk = sitk.ReadImage(mask_path)
        mask = sitk.GetArrayFromImage(mask_sitk)

        point_sitk = sitk.ReadImage(point_path)
        point = sitk.GetArrayFromImage(point_sitk)

        if self.config['gaussian']:
            ball = np.load(gausmap_path)
            # ballcluster = np.load(cluster_gausmap_path)
        else:
            distmap = np.load(distmap_path)
            ball = distmap <= self.config['dist_thr']
            # ballcluster_sitk = sitk.ReadImage(cluster_ball_path)
            # ballcluster = sitk.GetArrayFromImage(ballcluster_sitk)

        if self.config['atlas']:
            atlas = sitk.GetArrayFromImage(sitk.ReadImage('DATA/resize/ch2_rpi.nii'))
            if self.config['gaussian']:
                atlas_ball = np.load('DATA/resize/ch2_gaussian_rpi.npy')
            else:
                atlas_distmap = np.load('DATA/resize/ch2_distance_rpi.npy')
                atlas_ball = atlas_distmap <= self.config['dist_thr']

        # c = image.shape[0]
        # plt.imshow(atlas[c // 2])
        # plt.show()
        # plt.imshow(atlas_ball.sum(0)[c // 2])
        # plt.show()

        # if self.transform and random.random() <= 0.5:
        #     image = image[:, :, ::-1].copy()
        #     mask = mask[:, :, ::-1].copy()
        #     point = point[:, :, ::-1].copy()
        #     ball = ball[:, :, :, ::-1].copy()
        #     if self.config['atlas']:
        #         atlas = atlas[:, :, ::-1].copy()
        #         atlas_ball = atlas_ball[:, :, :, ::-1].copy()

        # plt.imshow(atlas[c // 2])
        # plt.show()
        # plt.imshow(atlas_ball.sum(0)[c // 2])
        # plt.show()

        image = torch.from_numpy(image).unsqueeze(0).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        point = torch.from_numpy(point).unsqueeze(0).float()

        # ballcluster = torch.from_numpy(ballcluster).float()
        ball = torch.from_numpy(ball).float()

        datadict = {
            'image': image,
            'mask': mask,
            'point': point,
            'ball': ball,
            'path': img_path,
            # 'ballcluster': ballcluster
        }

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
    is_segmentation = loader_config.get('is_segmentation', False)
    if is_segmentation:
        train_subjects_dataset = SegmentDataSet(loader_config, train_name_list, transform=False)
        val_subjects_dataset = SegmentDataSet(loader_config, val_name_list)
    else:
        train_subjects_dataset = KeyPointDataSet(loader_config, train_name_list, transform=False)
        val_subjects_dataset = KeyPointDataSet(loader_config, val_name_list)
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
