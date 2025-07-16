# full assembly of the sub-parts to form the complete net
# H W D process same
import math

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

# from .unet_parts import *
# from unet_parts import *
from patchify import patchify
from timm.models.layers import DropPath



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


class UNeXt3D_ablation(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNeXt3D_ablation, self).__init__()
        # input converlution: ?x32x32x32x1 ==> ?x32x32x32x32
        self.dow_conv1 = inconv(n_channels, 16)
        self.dow_conv2 = inconv(16, 32)
        self.dow_conv3 = inconv(32, 64)
        self.dow_conv4 = inconv(64, 128)
        self.dow_conv5 = inconv(128, 256)

        self.Bn_dow_conv1 = nn.BatchNorm3d(16)
        self.Bn_dow_conv2 = nn.BatchNorm3d(32)
        self.Bn_dow_conv3 = nn.BatchNorm3d(64)
        self.Bn_dow_conv4 = nn.BatchNorm3d(128)
        self.Bn_dow_conv5 = nn.BatchNorm3d(256)

        self.norm1 = nn.LayerNorm(128)
        self.norm2 = nn.LayerNorm(256)
        self.norm3 = nn.LayerNorm(256)

        self.decoder1 = nn.Conv3d(256, 128, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv3d(128, 64, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv3d(32, 16, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv3d(16, 4, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm3d(128)
        self.dbn2 = nn.BatchNorm3d(64)
        self.dbn3 = nn.BatchNorm3d(32)
        self.dbn4 = nn.BatchNorm3d(16)
        self.dbn5 = nn.BatchNorm3d(4)

        self.final = nn.Conv3d(4, n_classes, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]  # ？ 1 H W D
        # input converlution stage
        out = F.relu6(F.max_pool3d(self.Bn_dow_conv1(self.dow_conv1(x)), 2, 2))
        t1 = out  # ? 16 H/2 W/2 D/2
        out = F.relu6(F.max_pool3d(self.Bn_dow_conv2(self.dow_conv2(out)), 2, 2))
        t2 = out  # ? 32 H/4 W/4 D/4
        out = F.relu6(F.max_pool3d(self.Bn_dow_conv3(self.dow_conv3(out)), 2, 2))
        t3 = out  # ? 64 H/8 W/8 D/8
        out = F.relu6(F.max_pool3d(self.Bn_dow_conv4(self.dow_conv4(out)), 2, 2))
        t4 = out
        out = F.relu6(F.max_pool3d(self.Bn_dow_conv5(self.dow_conv5(out)), 2, 2))
        t5 = out
        ######################################

        #######################################################
        out = F.relu6(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t4)
        out = F.relu6(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t3)
        out = F.relu6(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t2)
        out = F.relu6(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t1)
        #######################################################
        out = F.relu6(F.interpolate(self.dbn5(self.decoder5(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        out = self.final(out)  # ? 2 H W D

        return out

if __name__=="__main__":
    input_data = torch.randn(1,1,128,128,128)
    model = UNeXt3D_ablation(1,2)
    output_data= model(input_data)
    print(output_data.shape)
