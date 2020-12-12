import torch
import torch.nn as nn
import torch.nn.functional as F
from core.models.buildingblocks import ExtResNetBlock, Upsampling, conv3d, SingleConv, ExtResNetBlock


class Hourglass(nn.Module):
    def __init__(self, recursive_nums, f, basic_module=ExtResNetBlock,
                 increase=0, conv_kernel_size=3,
                 layer_order='gcr', num_groups=8, padding=1):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = basic_module(f, f,
                                kernel_size=conv_kernel_size,
                                order=layer_order,
                                num_groups=num_groups,
                                padding=padding)
        # Lower branch
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        self.low1 = basic_module(f, nf,
                                 kernel_size=conv_kernel_size,
                                 order=layer_order,
                                 num_groups=num_groups,
                                 padding=padding)
        self.n = recursive_nums

        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(recursive_nums - 1, nf, basic_module=basic_module, increase=0,
                                  conv_kernel_size=conv_kernel_size,
                                  layer_order=layer_order,
                                  num_groups=num_groups,
                                  padding=padding)
        else:
            self.low2 = basic_module(nf, nf,
                                     kernel_size=conv_kernel_size,
                                     order=layer_order,
                                     num_groups=num_groups,
                                     padding=padding)

        self.low3 = basic_module(nf, nf,
                                 kernel_size=conv_kernel_size,
                                 order=layer_order,
                                 num_groups=num_groups,
                                 padding=padding)

        # self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2 = Upsampling(transposed_conv=False, mode='nearest')

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(up1, low3)
        return up1 + up2


class StackHourglass(nn.Module):
    def __init__(self, stack_nums, in_channels, out_channels, layer_order, num_groups,recursive_nums, final_sigmoid=False, **kwargs):
        super(StackHourglass, self).__init__()

        self.stack_nums = stack_nums
        self.testing = False

        self.pre = nn.Sequential(
            SingleConv(1, in_channels, 7, order='cgr', padding=3, num_groups=num_groups),
            ExtResNetBlock(in_channels, in_channels, order=layer_order, num_groups=num_groups),
            # nn.MaxPool3d(kernel_size=2),
            # ExtResNetBlock(128, 128, order=layer_order),
            # ExtResNetBlock(64, in_channels, order=layer_order)
        )

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(recursive_nums, in_channels,
                          basic_module=ExtResNetBlock,
                          conv_kernel_size=3,
                          layer_order=layer_order,
                          num_groups=num_groups),
            ) for _ in range(self.stack_nums)])

        self.features = nn.ModuleList([
            nn.Sequential(
                ExtResNetBlock(in_channels, in_channels, order=layer_order, num_groups=num_groups),
                SingleConv(in_channels, in_channels, 1, order=layer_order, padding=0, num_groups=num_groups),
                # Conv(in_channels, in_channels, 1, bn=True, relu=True)
            ) for _ in range(self.stack_nums)])

        self.merge_features = nn.ModuleList([conv3d(in_channels, in_channels, 1, bias=False, padding=0) for _ in range(self.stack_nums - 1)])
        self.outs = nn.ModuleList([conv3d(in_channels, out_channels, 1, bias=False, padding=0) for _ in range(self.stack_nums)])
        self.merge_preds = nn.ModuleList([conv3d(out_channels, in_channels, 1, bias=False, padding=0) for _ in range(self.stack_nums - 1)])

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, imgs):
        x = self.pre(imgs)
        combined_hm_preds = []
        for i in range(self.stack_nums):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.stack_nums - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        x = combined_hm_preds[-1]
        return x
    
    def keypoint_postprocess(self,x):
        # final_shape = x.shape
        # x = x.view(final_shape[0], final_shape[1], -1)
        # x = F.softmax(x, dim=2)
        # x = x.view(*final_shape)

        x = F.softmax(x, dim=1)
        return x

    def segment_postprocess(self,x):
        # final_shape = x.shape
        # x = x.view(final_shape[0], final_shape[1], -1)
        x = F.softmax(x, dim=1)
        # x = x.view(*final_shape)
        return x


if __name__ == '__main__':
    a = torch.zeros((1, 1, 64, 64, 64))
    model = StackHourglass(2, 32, 17, 'gcr', 8)
    b = model(a)
    print(b.shape)
