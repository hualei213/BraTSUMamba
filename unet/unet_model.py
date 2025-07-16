# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from .unet_parts import *


class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet3D, self).__init__()
        # input converlution: ?x32x32x32x1 ==> ?x32x32x32x32
        self.inc = inconv(n_channels, 32)

        # down layer0: ?x32x32x32x32 ==> ?x32x32x32x32
        self.down0 = down(in_ch=32, out_ch=32, strides=1)
        # down layer1: ?x32x32x32x32 ==> ?x16x16x16x64
        self.down1 = down(in_ch=32, out_ch=64, strides=2)

        # bottom layer: ?x16x16x16x64 ==> ?x8x8x8x128
        self.down2 = down(in_ch=64, out_ch=128, strides=2)

        # up layer0: ?x8x8x8x128 ==> ?x16x16x16x64
        self.up0 = up(128, 64)
        # up layer1: ?x16x16x16x64 ==> ?x32x32x32x32
        self.up1 = up(64, 32)

        # output layer: ?x32x32x32x32 ==> ?x32x32x32x4
        self.outc = outconv(32, n_classes)

    def forward(self, x):
        # input converlution layer
        x0 = self.inc(x=x)
        # down layers
        x1 = self.down0(x=x0)
        x2 = self.down1(x=x1)
        x3 = self.down2(x=x2)
        # up layers
        x4 = self.up0(x=x3, skip_connct=x2)
        x5 = self.up1(x=x4, skip_connct=x1)
        # output layer
        x6 = self.outc(x5)

        return x6
