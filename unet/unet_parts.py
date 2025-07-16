# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


class inconv(nn.Module):
    """
    input converlution layer
    ?x32x32x32x1 ==> ?x32x32x32x32
    """

    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch,
                              out_channels=out_ch,
                              kernel_size=3,
                              padding=1,
                              stride=1,
                              bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv_5(nn.Module):
    """
    input converlution layer
    ?x32x32x32x1 ==> ?x32x32x32x32
    """

    def __init__(self, in_ch, out_ch):
        super(inconv_5, self).__init__()
        self.conv_5 = nn.Conv3d(in_channels=in_ch,
                              out_channels=out_ch,
                              kernel_size=5,
                              padding=2,
                              stride=1,
                              bias=False)

    def forward(self, x):
        x = self.conv_5(x)
        return x

class inconv_7(nn.Module):
    """
    input converlution layer
    ?x32x32x32x1 ==> ?x32x32x32x32
    """

    def __init__(self, in_ch, out_ch):
        super(inconv_7, self).__init__()
        self.conv_7 = nn.Conv3d(in_channels=in_ch,
                              out_channels=out_ch,
                              kernel_size=7,
                              padding=3,
                              stride=1,
                              bias=False)

    def forward(self, x):
        x = self.conv_7(x)
        return x


class down(nn.Module):
    """
    down layer
    """

    def __init__(self, in_ch, out_ch, strides):
        super(down, self).__init__()
        self.layers = nn.Sequential(
            residual_block_type_0(in_ch=in_ch, out_ch=out_ch, strides=strides),
            residual_block_type_1(in_ch=out_ch, out_ch=out_ch)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class up(nn.Module):
    """
    up layer
    """

    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.convTranspose = nn.ConvTranspose3d(in_channels=in_ch,
                                                out_channels=out_ch,
                                                kernel_size=2,
                                                stride=2,
                                                bias=False
                                                )
        self.residual = residual_block_type_1(in_ch=out_ch, out_ch=out_ch)

    def forward(self, x, skip_connct):
        x1 = self.convTranspose(x)
        x2 = x1 + skip_connct
        x3 = self.residual(x2)
        return x3


class outconv(nn.Module):
    """
    output converlution layer
    """

    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm3d(num_features=in_ch, momentum=0.997),
            nn.ReLU6(),
            nn.Dropout(),
            nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.layers(x)
        return x1


class residual_block_type_0(nn.Module):
    """
    residual block, type 0
    """

    def __init__(self, in_ch, out_ch, strides):
        super(residual_block_type_0, self).__init__()

        self.BN_Relu1 = nn.Sequential(
            nn.BatchNorm3d(num_features=in_ch, momentum=0.997),
            nn.ReLU6()
        )
        self.conv1 = nn.Conv3d(in_channels=in_ch,
                               out_channels=out_ch,
                               kernel_size=1,
                               stride=strides,
                               padding=0,
                               bias=False)
        self.conv3 = nn.Conv3d(in_channels=in_ch,
                               out_channels=out_ch,
                               kernel_size=3,
                               stride=strides,
                               padding=1,
                               bias=False)
        self.BN_Relu2 = nn.Sequential(
            nn.BatchNorm3d(num_features=out_ch, momentum=0.997),
            nn.ReLU6())
        self.conv31 = nn.Conv3d(in_channels=out_ch,
                                out_channels=out_ch,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)

    def forward(self, x):
        shortcut = self.conv1(x)
        x1 = self.BN_Relu1(x)
        x2 = self.conv3(x1)
        x3 = self.BN_Relu2(x2)
        x4 = self.conv31(x3)

        return x4 + shortcut


class residual_block_type_1(nn.Module):
    """
    residual block, type 1
    """

    def __init__(self, in_ch, out_ch):
        super(residual_block_type_1, self).__init__()

        self.BN_Relu1 = nn.Sequential(
            nn.BatchNorm3d(num_features=in_ch, momentum=0.997),
            nn.ReLU6()
        )
        self.conv31 = nn.Conv3d(in_channels=in_ch,
                                out_channels=out_ch,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.BN_Relu2 = nn.Sequential(
            nn.BatchNorm3d(num_features=out_ch, momentum=0.997),
            nn.ReLU6()
        )
        self.conv32 = nn.Conv3d(in_channels=out_ch,
                                out_channels=out_ch,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)

    def forward(self, x):
        shortcut = x
        x1 = self.BN_Relu1(x)
        x2 = self.conv31(x1)
        x3 = self.BN_Relu2(x2)
        x4 = self.conv32(x3)

        return x4 + shortcut
