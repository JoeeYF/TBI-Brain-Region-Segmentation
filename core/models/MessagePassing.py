import torch
import torch.nn as nn
import numpy as np


def conv3d(in_channels, out_channels, kernel_size, padding, bias=True):
    conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
    data = np.zeros((kernel_size, kernel_size, kernel_size))
    data[kernel_size//2, kernel_size//2, kernel_size//2] = 1
    conv.weight.data = nn.Parameter(torch.from_numpy(data), requires_grad=True)
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


class MessagePassing(nn.Module):
    def __init__(self, direction=True):
        super(MessagePassing, self).__init__()
        self.direction = direction
        bias = False
        self.conv01 = conv3d(1, 1, 3, 1, bias)
        self.conv02 = conv3d(1, 1, 3, 1, bias)
        self.conv20 = conv3d(1, 1, 3, 1, bias)
        self.conv50 = conv3d(1, 1, 3, 1, bias)

        self.conv10 = conv3d(1, 1, 3, 1, bias)
        self.conv11 = conv3d(1, 1, 3, 1, bias)
        self.conv30 = conv3d(1, 1, 3, 1, bias)
        self.conv70 = conv3d(1, 1, 3, 1, bias)

        self.conv80 = conv3d(1, 1, 3,  1,  bias)
        self.conv100 = conv3d(1, 1,3,  1,  bias)
        self.conv120 = conv3d(1, 1,3,  1,  bias)
        self.conv130 = conv3d(1, 1,3,  1,  bias)

        # self.conv90 = conv3d(1, 1, 7,   3, bias)
        # self.conv110 = conv3d(1, 1, 7,  3, bias)
        # self.conv8_12 = conv3d(1, 1, 7,   3, bias)
        # self.conv12_13 = conv3d(1, 1, 7,  3, bias)
        # self.conv10_13 = conv3d(1, 1, 7,  3, bias)
        # self.conv4_0 = conv3d(1, 1, 7, 3,    bias)
        # self.conv6_1 = conv3d(1, 1, 7, 3,    bias)

    def forward(self, feature):
        feature_list = [feature[:, i, ...].unsqueeze(1) for i in range(17)]

        feature_list[0] = feature_list[0] + self.conv01(feature_list[5]) + self.conv02(feature_list[4])
        feature_list[5] = feature_list[5] + self.conv50(feature_list[4])
        feature_list[2] = feature_list[2] + self.conv20(feature_list[5])

        feature_list[1] = feature_list[1] + self.conv10(feature_list[7]) + self.conv11(feature_list[6])
        feature_list[7] = feature_list[7] + self.conv70(feature_list[6])
        feature_list[3] = feature_list[3] + self.conv30(feature_list[7])
        
        feature_list[8] = feature_list[8] + self.conv80(feature_list[10])
        feature_list[10] = feature_list[10] + self.conv100(feature_list[8])
        feature_list[12] = feature_list[12] + self.conv120(feature_list[8])
        feature_list[13] = feature_list[13] + self.conv130(feature_list[10])

        # feature_list[9] = feature_list[9] + self.conv90(feature_list[16])
        # feature_list[11] = feature_list[11] + self.conv110(feature_list[16])
        return torch.cat(feature_list, dim=1)
