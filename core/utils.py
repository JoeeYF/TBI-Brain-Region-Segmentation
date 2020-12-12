import logging
import os
import shutil
import sys
import yaml
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
import SimpleITK as sitk

# plt.ioff()
# plt.switch_backend('agg')


def load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))


def increment_dir(dir, e=None):
    # Increments a directory runs/exp1 --> runs/exp2_comment
    n = 0  # number
    d = sorted(glob(dir + '[0-9]*'))  # directories
    if len(d):
        n = int(d[-1].split('/')[-1].split('-')[0][-3:]) + 1  # increment
    if e is not None:
        os.makedirs(dir + str(n).zfill(3) + '-' + e, exist_ok=True)
        return dir + str(n).zfill(3) + '-'+e
    else:
        os.makedirs(dir + '-' + str(n).zfill(3), exist_ok=True)
        return dir + str(n).zfill(3) + '-'


class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.value = value
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, checkpoint_dir, logger=None):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.

    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        shutil.copyfile(last_file_path, best_file_path)


def save_nii(array, reference_name, path):
    img_ref = sitk.ReadImage(reference_name)
    img_itk = sitk.GetImageFromArray(array.astype(sitk.GetArrayFromImage(img_ref).dtype))
    img_itk.CopyInformation(img_ref)
    sitk.WriteImage(img_itk, path)

def cal_dice(pred, label):
    assert(len(pred.shape) == 3)
    assert(pred.shape == label.shape)
    dice_list = []
    for i in range(1, 18):
        dice = binary_dice3d(pred == i, label == i)
        dice_list.append(dice)
    return np.mean(dice_list), dice_list

def binary_dice3d(s, g):
    """
    dice score of 3d binary volumes
    inputs: 
        s: segmentation volume
        g: ground truth volume
    outputs:
        dice: the dice score
    """
    assert(len(s.shape) == 3)
    [Ds, Hs, Ws] = s.shape
    [Dg, Hg, Wg] = g.shape
    assert(Ds == Dg and Hs == Hg and Ws == Wg)
    prod = np.multiply(s, g)
    s0 = prod.sum()
    s1 = s.sum()
    s2 = g.sum()
    dice = (2.0*s0 + 1e-10)/(s1 + s2 + 1e-10)
    return dice
loggers = {}


def get_logger(name, logFilename='/home/yuanfang/pytorch3dunet/work_result/test.log', level=logging.INFO):
    global loggers

    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        formatter = logging.Formatter('[%(asctime)s-%(levelname)s-%(name)s] %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        # Logging to file
        fh = logging.FileHandler(logFilename, mode='a')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        loggers[name] = logger
        return logger


def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW)
    """
    if input.dim() == 4:
        # expand the input tensor to Nx1xDxHxW before scattering
        input = input.unsqueeze(1)

    # create result tensor shape (NxCxDxHxW)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input.long(), 1)


def get_distance_map(point):
    assert len(point.shape) == 3
    distmap_list = []
    for i in range(1, 18):
        pointcoord = np.where(point == i)
        coord = np.array(np.where(point < 1e10))
        coord = np.stack(coord, axis=1).reshape(point.shape[0], point.shape[1], point.shape[2], 3)
        distmap = np.sqrt(((coord - np.array([pointcoord[0][0], pointcoord[1][0], pointcoord[2][0]])) ** 2).sum(-1))
        distmap_list.append(distmap)
    return np.array(distmap_list)


def get_distance_2point(point1, point2):
    assert point1.size() == point2.size()
    dis = ((point1-point2)**2).sum().sqrt()
    return dis


# def find_maximum_patch_size(model, device):
#     """Tries to find the biggest patch size that can be send to GPU for inference
#     without throwing CUDA out of memory"""
#     logger = get_logger('PatchFinder')
#     in_channels = model.in_channels

#     patch_shapes = [(64, 128, 128), (96, 128, 128),
#                     (64, 160, 160), (96, 160, 160),
#                     (64, 192, 192), (96, 192, 192)]

#     for shape in patch_shapes:
#         # generate random patch of a given size
#         patch = np.random.randn(*shape).astype('float32')

#         patch = torch \
#             .from_numpy(patch) \
#             .view((1, in_channels) + patch.shape) \
#             .to(device)

#         logger.info(f"Current patch size: {shape}")
#         model(patch)


# def remove_halo(patch, index, shape, patch_halo):
#     """
#     Remove `pad_width` voxels around the edges of a given patch.
#     """
#     assert len(patch_halo) == 3

#     def _new_slices(slicing, max_size, pad):
#         if slicing.start == 0:
#             p_start = 0
#             i_start = 0
#         else:
#             p_start = pad
#             i_start = slicing.start + pad

#         if slicing.stop == max_size:
#             p_stop = None
#             i_stop = max_size
#         else:
#             p_stop = -pad if pad != 0 else 1
#             i_stop = slicing.stop - pad

#         return slice(p_start, p_stop), slice(i_start, i_stop)

#     D, H, W = shape

#     i_c, i_z, i_y, i_x = index
#     p_c = slice(0, patch.shape[0])

#     p_z, i_z = _new_slices(i_z, D, patch_halo[0])
#     p_y, i_y = _new_slices(i_y, H, patch_halo[1])
#     p_x, i_x = _new_slices(i_x, W, patch_halo[2])

#     patch_index = (p_c, p_z, p_y, p_x)
#     index = (i_c, i_z, i_y, i_x)
#     return patch[patch_index], index
