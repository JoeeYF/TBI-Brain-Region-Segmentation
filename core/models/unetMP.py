import torch.nn as nn
import torch.nn.functional as F
import torch

from .buildingblocks import Encoder, Decoder, DoubleConv, ExtResNetBlock
from .MessagePassing import MessagePassing

from core.utils import number_of_features_per_level


class MPB(nn.Module):
    def __init__(self, in_channels):
        super(MPB, self).__init__()
        self.conv_heatmap = nn.Sequential(
            # ExtResNetBlock(in_channels, in_channels),
            nn.Conv3d(in_channels, 17, 3, padding=1, bias=False)
        )
        self.conv_heatmapout = nn.Sequential(
            # ExtResNetBlock(in_channels, in_channels),
            nn.Conv3d(17, in_channels, 3, padding=1, bias=False)
        )
        self.messagepass = MessagePassing()
        # self.mlp = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(17, in_channels // 2),
        #     nn.ReLU(),
        #     nn.Linear(in_channels // 2, in_channels)
        # )

    def forward(self, x):
        heatmap = self.conv_heatmap(x)
        heatmap_mp = self.messagepass(heatmap)
        heatmap_mp = heatmap + heatmap_mp

        ## spital attention
        heatmap_mp_out = self.conv_heatmapout(heatmap_mp)

        ## channel attention
        avg_pool = F.avg_pool3d(heatmap_mp, (heatmap_mp.size(2), heatmap_mp.size(3), heatmap_mp.size(4)),
                                stride=(heatmap_mp.size(2), heatmap_mp.size(3), heatmap_mp.size(4)))
        max_pool = F.max_pool3d(heatmap_mp, (heatmap_mp.size(2), heatmap_mp.size(3), heatmap_mp.size(4)),
                                stride=(heatmap_mp.size(2), heatmap_mp.size(3), heatmap_mp.size(4)))
        channel_att_raw = self.mlp(avg_pool) + self.mlp(max_pool)
        channel_att_raw = channel_att_raw.view(channel_att_raw.size(0), channel_att_raw.size(1), 1, 1, 1)
        # channel_att_raw = torch.sigmoid(channel_att_raw)

        heatmap_mp_x = x * torch.sigmoid(channel_att_raw)*torch.sigmoid(heatmap_mp_out)
        return F.softmax(heatmap_mp, dim=1), heatmap_mp_x + x


class Abstract3DUNet(nn.Module):

    def __init__(self, in_channels, out_channels, basic_module,
                 f_maps=[16, 32, 64], layer_order='gcr', num_groups=8,
                 conv_kernel_size=3, pool_kernel_size=2, conv_padding=1, mpb_nums=1, **kwargs):

        super(Abstract3DUNet, self).__init__()

        encoders = []
        encodermpbs = []

        if mpb_nums == 1:
            self.encoder_mpb_pos = [0]
            self.decoder_mpb_pos = [len(f_maps) - 2]
        elif mpb_nums == 2:
            self.encoder_mpb_pos = [0, 1]
            self.decoder_mpb_pos = [len(f_maps) - 3, len(f_maps) - 2]

        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num,
                                  basic_module=basic_module,
                                  conv_layer_order=layer_order,
                                  conv_kernel_size=conv_kernel_size,
                                  num_groups=num_groups,
                                  padding=conv_padding,
                                  apply_pooling=False)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num,
                                  basic_module=basic_module,
                                  conv_layer_order=layer_order,
                                  conv_kernel_size=conv_kernel_size,
                                  num_groups=num_groups,
                                  padding=conv_padding,
                                  apply_pooling=True,
                                  pool_kernel_size=pool_kernel_size)
            if i in self.encoder_mpb_pos:
                mpb = MPB(out_feature_num)
                encodermpbs.append(mpb)

            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)
        self.encodermpbs = nn.ModuleList(encodermpbs)

        decoders = []
        decodermpbs = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            if basic_module == DoubleConv:
                in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            else:
                in_feature_num = reversed_f_maps[i]

            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num,
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              padding=conv_padding)
            decoders.append(decoder)
            if i in self.decoder_mpb_pos:
                mpb = MPB(out_feature_num)
                decodermpbs.append(mpb)

        self.decoders = nn.ModuleList(decoders)
        self.decodermpbs = nn.ModuleList(decodermpbs)
        # self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 3, 1, 1)

    def forward(self, x):
        encoders_features = []
        encoders_heatmaps = []
        decoders_heatmaps = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            encoders_features.insert(0, x)
            if i in self.encoder_mpb_pos:
                heatmap, x = self.encodermpbs[i](x)
                encoders_heatmaps.append(heatmap)

        for i, (decoder, encoder_features) in enumerate(zip(self.decoders, encoders_features[1:])):
            x = decoder(encoder_features, x)
            if i in self.decoder_mpb_pos:
                heatmap, x = self.decodermpbs[i - len(self.decoders)](x)
                decoders_heatmaps.append(heatmap)

        x = self.final_conv(x)
        x = F.softmax(x, dim=1)
        return encoders_heatmaps, decoders_heatmaps, x


class ResidualMPUNet3D(Abstract3DUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ExtResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels,
                 f_maps=[16, 32, 64], layer_order='gcr', num_groups=8,
                 conv_padding=1, **kwargs):
        super(ResidualMPUNet3D, self).__init__(in_channels=in_channels, out_channels=out_channels, basic_module=ExtResNetBlock,
                                               f_maps=f_maps, layer_order=layer_order, num_groups=num_groups,
                                               conv_kernel_size=3, pool_kernel_size=2, conv_padding=conv_padding,
                                               **kwargs)
