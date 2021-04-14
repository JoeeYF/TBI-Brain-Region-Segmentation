import torch
import torch.nn as nn
import numpy as np


def conv3d(in_channels, out_channels, kernel_size, dilation=1, bias=False):
    # conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
    # data = np.zeros((kernel_size, kernel_size, kernel_size))
    # data[kernel_size // 2, kernel_size // 2, kernel_size // 2] = 1
    # conv.weight.data = nn.Parameter(torch.from_numpy(data), requires_grad=True)
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1) - 1) // 2
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias, dilation=dilation)


class MessagePassing(nn.Module):
    def __init__(self, direction=True):
        super(MessagePassing, self).__init__()
        self.direction = direction
        bias = False
        self.conv04 = conv3d(1, 1, 7)
        self.conv05 = conv3d(1, 1, 7)
        self.conv52 = conv3d(1, 1, 3)
        self.conv24 = conv3d(1, 1, 7)

        self.conv16 = conv3d(1, 1, 7)
        self.conv17 = conv3d(1, 1, 7)
        self.conv73 = conv3d(1, 1, 3)
        self.conv36 = conv3d(1, 1, 7)

        self.conv29 = conv3d(1, 1, 7, dilation=2, bias=bias)
        self.conv311 = conv3d(1, 1, 7, dilation=2, bias=bias)
        # self.conv29_1 = conv3d(1, 1, 7, bias)
        # self.conv311_1 = conv3d(1, 1, 7, bias)

        self.conv80 = conv3d(1, 1, 3)
        self.conv100 = conv3d(1, 1, 3)
        self.conv120 = conv3d(1, 1, 7)
        self.conv130 = conv3d(1, 1, 7)

    def forward(self, feature):
        feature_list = [feature[:, i, ...].unsqueeze(1) for i in range(17)]

        feature_list[4] = feature_list[4] + self.conv04(feature_list[0])
        feature_list[5] = feature_list[5] + self.conv05(feature_list[0])
        feature_list[2] = feature_list[2] + self.conv52(feature_list[5])
        feature_list[4] = feature_list[4] + self.conv24(feature_list[2])

        feature_list[6] = feature_list[6] + self.conv16(feature_list[1])
        feature_list[7] = feature_list[7] + self.conv17(feature_list[1])
        feature_list[3] = feature_list[3] + self.conv73(feature_list[7])
        feature_list[6] = feature_list[6] + self.conv36(feature_list[3])

        feature_list[9] = feature_list[9] + self.conv29(feature_list[2])
        feature_list[11] = feature_list[11] + self.conv311(feature_list[3])

        feature_list[8] = feature_list[8] + self.conv80(feature_list[10])
        feature_list[10] = feature_list[10] + self.conv100(feature_list[8])
        feature_list[12] = feature_list[12] + self.conv120(feature_list[8])
        feature_list[13] = feature_list[13] + self.conv130(feature_list[10])

        return torch.cat(feature_list, dim=1)
