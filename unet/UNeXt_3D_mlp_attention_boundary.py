# full assembly of the sub-parts to form the complete net
# H W D process same


import math

import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from unet.unet_parts import *
from patchify import patchify
from timm.models.layers import DropPath

from utils import get_sobel, run_sobel, run_sobel2gt

"""
在hierarchical MLP的基础上在网络的前3层增加MLP attention的操作
这里的MLP使用的是UNeXt的MLP结构
增加边界注意力模块
"""


#############新增加分支内容

class MLP_attention(nn.Module):
    def __init__(self, inchannle=3):
        super(MLP_attention, self).__init__()

    def forward(self, out_cnn, out_mlp):
        a = F.softmax(out_mlp)
        residual_cnn = out_cnn
        out = out_cnn * a
        out = out + residual_cnn
        return out


class unext_shift_mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv1 = DWConv(hidden_features)
        self.dwconv2 = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.fc3 = nn.Linear(hidden_features, out_features)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def shift(self, x, H, W, D, axis):
        B, N, C = x.shape
        xn = x.transpose(1, 2).view(B, C, H, W, D).contiguous()
        # shift domain
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)  # seplit xn along 1 axis
        x_shift = [torch.roll(x_c, shift, axis) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        # tokenize domain
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_cat = torch.narrow(x_cat, 3, self.pad, W)
        x_s = torch.narrow(x_cat, 4, self.pad, D)
        # tokens doamin
        x_s = x_s.reshape(B, C, H * W * D).contiguous()
        x = x_s.transpose(1, 2)
        return x

    def forward(self, x, H, W, D):
        """
            x (B,C,H,W,D)
        """
        x = self.shift(x, H, W, D, 2)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.dwconv1(x, H, W, D)

        x = self.shift(x, H, W, D, 3)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.dwconv2(x, H, W, D)

        x = self.shift(x, H, W, D, 4)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.act(x)  # GELU

        return x


class unext_shifted_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=1., drop=0.2, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.drop_path = DropPath(drop) if drop > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = unext_shift_mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, D):
        """
        input shape (B,N,C)
        output shape (B,N,C)
        """
        # H W D axis shift
        H_W_D_Shift = self.mlp(self.norm2(x), H, W, D)
        x = x + self.drop_path(H_W_D_Shift)
        return x


class MLP(torch.nn.Module):

    def __init__(self, num_i, num_o):
        super(MLP, self).__init__()
        self.proj = OverlapPatchEmbed(patch_size=3, in_chans=num_i, embed_dim=num_o)
        self.block = unext_shifted_Block(dim=num_o, mlp_ratio=1)
        self.norm = nn.LayerNorm(num_o)

    def forward(self, x):
        B, _, _, _, _ = x.shape
        # 输入影像首先做flatten
        out, H, W, D = self.proj(x)
        # MLP stage
        out = self.block(out, H, W, D)
        out = self.norm(out)
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        return out


class boundary_attention(nn.Module):
    # sobel 算子 https://github.com/HXY-99/brats/blob/main/model/edge.py
    def __init__(self, in_channels):
        super(boundary_attention, self).__init__()
        self.sobel_x1, self.sobel_y1, self.sobel_z1 = get_sobel(in_channels, 1)
        self.conv_e_1 = nn.Conv3d(in_channels, in_channels, 3, 1, 1)
        self.bn_e_1 = nn.BatchNorm3d(in_channels)

        self.conv_d_1 = nn.Conv3d(in_channels, in_channels, 3, 1, 1)
        self.bn_d_1 = nn.BatchNorm3d(in_channels)

        self.conv_out_1 = nn.Conv3d(in_channels, in_channels, 1, 1, 0)

    def forward(self, boundary_image, guided_image):
        """
            parameter boundary_image: 边界图像
            parameter guided_image:被引导图像
        """
        #### encoder stage
        sobel_x = run_sobel(self.sobel_x1, self.sobel_y1, self.sobel_z1, boundary_image)
        sobel_x = F.relu(self.bn_e_1(self.conv_e_1(sobel_x))) + boundary_image
        sobel_x = F.sigmoid(sobel_x)
        #### decoder stage
        guided_image = F.relu(self.bn_d_1(self.conv_d_1(guided_image)))
        out = sobel_x * guided_image
        out = out + sobel_x
        out = self.conv_out_1(out)
        out = F.relu(out)
        return out


##############以上为新增加的分支内容
class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=2, stride=2, in_chans=3, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        """
        input shape (B,C,H,W,D)
        output shape (B,N,C)
        """
        x = self.proj(x)
        _, _, H, W, D = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W, D


class DWConv(nn.Module):
    def __init__(self, dim=128):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W, D):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W, D)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.fc3 = nn.Linear(hidden_features, out_features)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        self.image_size = 128  # 输出的图像的大小

        # self.reshape_mlp_1 = nn.Linear((self.image_size//(2**1))**3*2, (self.image_size//(2**1))**3)  # 这是个超参数，H*W*D*2,H*W*D需要根据当前输入图片大小进行调整
        # self.reshape_mlp_2 = nn.Linear((self.image_size//(2**2))**3*2, (self.image_size//(2**2))**3)  # 这是个超参数，H*W*D*2,H*W*D需要根据当前输入图片大小进行调整
        # self.reshape_mlp_3 = nn.Linear((self.image_size//(2**3))**3*2, (self.image_size//(2**3))**3)  # 这是个超参数，H*W*D*2,H*W*D需要根据当前输入图片大小进行调整
        self.reshape_mlp_4 = nn.Linear((self.image_size // (2 ** 4)) ** 3 * 2,
                                       (self.image_size // (2 ** 4)) ** 3)  # 这是个超参数，H*W*D*2,H*W*D需要根据当前输入图片大小进行调整
        self.reshape_mlp_5 = nn.Linear((self.image_size // (2 ** 5)) ** 3 * 2,
                                       (self.image_size // (2 ** 5)) ** 3)  # 这是个超参数，需要根据当前输入图片大小进行调整

        self.single_mlp_1 = nn.Linear((self.image_size // (2 ** 4)) ** 3,
                                      (self.image_size // (2 ** 4)) ** 3)  # 这是个超参数，需要根据当前输入图片大小进行调整
        self.single_mlp_2 = nn.Linear((self.image_size // (2 ** 5)) ** 3,
                                      (self.image_size // (2 ** 5)) ** 3)  # 这是个超参数，需要根据当前输入图片大小进行调整
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def shift(self, x, H, W, D, axis, roll_rule):
        B, N, C = x.shape
        xn = x.transpose(1, 2).view(B, C, H, W, D).contiguous()
        # shift domain
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)  # seplit xn along 1 axis
        x_shift = [torch.roll(x_c, shift, axis) for x_c, shift in zip(xs, roll_rule)]
        # tokenize domain
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_cat = torch.narrow(x_cat, 3, self.pad, W)
        x_s = torch.narrow(x_cat, 4, self.pad, D)
        # tokens doamin
        x_s = x_s.reshape(B, C, H * W * D).contiguous()
        x = x_s
        return x

    def choose_mlp(self, x, H_size):
        if H_size == (self.image_size // (2 ** 1)):
            return self.reshape_mlp_1(x)
        if H_size == (self.image_size // (2 ** 2)):
            return self.reshape_mlp_2(x)
        if H_size == (self.image_size // (2 ** 3)):
            return self.reshape_mlp_3(x)
        if H_size == (self.image_size // (2 ** 4)):
            return self.reshape_mlp_4(x)
        if H_size == (self.image_size // (2 ** 5)):
            return self.reshape_mlp_5(x)

    def choose_single_mlp(self, x, H_size):
        if H_size == (self.image_size // (2 ** 4)):
            return self.single_mlp_1(x)
        if H_size == (self.image_size // (2 ** 5)):
            return self.single_mlp_2(x)

    def hierachical_mlp(self, x, H, W, D, axis):
        count = 3  # 自定义移动多少次IBSR:4  NFBS:3
        rule_list = [[-2, -1, 0, 1, 2],
                     [-1, 0, 1, 2, -2],
                     [0, 1, 2, -2, -1],
                     [1, 2, -2, -1, 0],
                     [2, -2, -1, 0, 1]]
        rule_list = rule_list[:count]
        result = self.shift(x, H, W, D, axis=axis, roll_rule=rule_list[0])
        for i in range(1, len(rule_list)):
            shift_1 = self.shift(x, H, W, D, axis=axis, roll_rule=rule_list[i])
            shift_1_1 = torch.concat([result, shift_1], dim=2)
            result = self.choose_mlp(shift_1_1, H)
        if len(rule_list) == 1:
            result = self.choose_single_mlp(result, H)
        return result.transpose(1, 2)

    def forward(self, x, H, W, D):
        """
            x (B,C,H,W,D)
        """
        x = self.dwconv(x, H, W, D)
        x = self.act(x)  # GELU
        x = self.hierachical_mlp(x, H, W, D, 2)
        x = self.drop(x)
        x = self.hierachical_mlp(x, H, W, D, 3)
        x = self.drop(x)
        x = self.hierachical_mlp(x, H, W, D, 4)
        x = self.drop(x)

        return x


class shiftedBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=1., drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, D):
        """
        input shape (B,N,C)
        output shape (B,N,C)
        """
        # H W D axis shift
        H_W_D_Shift = self.norm2(self.mlp(x, H, W, D))
        x = x + self.drop_path(H_W_D_Shift)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        # self.register_buffer()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SCA_module(nn.Module):
    def __init__(self, in_channels, out_channels,ratio):
        super(SCA_module, self).__init__()
        self.ca_conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.sa_conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.ca = ChannelAttention(in_planes=in_channels)
        self.sa = SpatialAttention(kernel_size=7)

        self.conv = nn.Conv3d(in_channels, 4, kernel_size=1)
        self.ratio = ratio
    def forward(self, x):
        residual_x = x
        ca_x = self.ca_conv(x)
        sa_x = self.sa_conv(x)
        ca_x = self.ca(ca_x) * ca_x
        sa_x = self.sa(sa_x) * sa_x
        fusion_x = ca_x * sa_x
        out = residual_x + fusion_x

        # 卷积和插值，变为原来的尺寸大小
        pred = self.conv(out)
        pred = F.interpolate(pred, scale_factor=self.ratio, mode='trilinear')
        return out,pred


class UNeXt3D_mlp_attention_boundary(nn.Module):
    def __init__(self, n_channels, n_classes, deep_flag=True):
        super(UNeXt3D_mlp_attention_boundary, self).__init__()

        # input converlution
        self.dow_conv1 = nn.Conv3d(in_channels=n_channels, out_channels=16, kernel_size=3, padding=1, stride=1,
                                   bias=False)
        self.dow_conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1, bias=False)
        self.dow_conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False)

        self.Bn_dow_conv1 = nn.BatchNorm3d(16)
        self.Bn_dow_conv2 = nn.BatchNorm3d(32)
        self.Bn_dow_conv3 = nn.BatchNorm3d(64)

        self.patch_embed1 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=64, embed_dim=128)
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=128, embed_dim=256)
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=1, in_chans=256, embed_dim=256)

        self.block1 = shiftedBlock(dim=128, mlp_ratio=1)
        self.block2 = shiftedBlock(dim=256, mlp_ratio=1)
        self.block3 = shiftedBlock(dim=256, mlp_ratio=1)

        self.dblock1 = shiftedBlock(dim=128, mlp_ratio=1)
        self.dblock2 = shiftedBlock(dim=64, mlp_ratio=1)

        self.norm1 = nn.LayerNorm(128)
        self.norm2 = nn.LayerNorm(256)
        self.norm3 = nn.LayerNorm(256)

        self.dnorm1 = nn.LayerNorm(128)
        self.dnorm2 = nn.LayerNorm(64)

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

        ###
        self.mlp1 = MLP(num_i=1, num_o=16)
        self.mlp2 = MLP(num_i=16, num_o=32)
        self.mlp3 = MLP(num_i=32, num_o=64)
        self.mlp_att1 = MLP_attention(16)
        self.mlp_att2 = MLP_attention(32)
        self.mlp_att3 = MLP_attention(64)
        self.ba4 = boundary_attention(128)
        self.ba3 = boundary_attention(64)
        self.ba2 = boundary_attention(32)
        self.ba1 = boundary_attention(16)
        ###
        ## 边缘loss，对每一层数据进行上采样插值
        self.deep_flag = deep_flag
        self.sobel_x1, self.sobel_y1, self.sobel_z1 = get_sobel(n_classes, 1)
        self.sca_5 = SCA_module(in_channels=128, out_channels=128,ratio=16)
        self.sca_4 = SCA_module(in_channels=64, out_channels=64,ratio=8)
        self.sca_3 = SCA_module(in_channels=32, out_channels=32,ratio=4)
        self.sca_2 = SCA_module(in_channels=16, out_channels=16,ratio=2)
        self.sca_1 = SCA_module(in_channels=4, out_channels=4,ratio=1)

    def forward(self, x):
        deep_preds = None
        B = x.shape[0]  # ？ 1 H W D
        # Tokenized MLP Stage
        mlp1 = self.mlp1(x)
        mlp2 = self.mlp2(mlp1)
        mlp3 = self.mlp3(mlp2)
        ##############

        # input converlution stage
        out = F.relu6(F.max_pool3d(self.Bn_dow_conv1(self.dow_conv1(x)), 2, 2))
        out = self.mlp_att1(out, mlp1)
        t1 = out  # ? 16 H/2 W/2 D/2

        out = F.relu6(F.max_pool3d(self.Bn_dow_conv2(self.dow_conv2(out)), 2, 2))
        out = self.mlp_att2(out, mlp2)
        t2 = out  # ? 32 H/4 W/4 D/4
        out = F.relu6(F.max_pool3d(self.Bn_dow_conv3(self.dow_conv3(out)), 2, 2))
        out = self.mlp_att3(out, mlp3)
        t3 = out  # ? 64 H/8 W/8 D/8

        ### Tokenized MLP Stage
        out, H, W, D = self.patch_embed1(out)  # ? N 128 N=H/16 w/16 D/16
        out = self.block1(out, H, W, D)  # ? N 128
        out = self.norm1(out)  # ? N 128
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        t4 = out  # ? 128 H/16 W/16 D/16

        # buttom block
        out, H, W, D = self.patch_embed2(out)  # ? N 256
        out = self.block2(out, H, W, D)  # ? N 256
        out = self.norm2(out)
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()  # ? 256 H/32 W/32 D/32

        out, H, W, D = self.patch_embed3(out)  # ? N 256
        out = self.block3(out, H, W, D)  # ? N 256
        out = self.norm3(out)
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()  # ? 256 H/32 W/32 D/32

        ### up stage
        # up1
        out = self.dbn1(self.decoder1(out))  # ? 128 H/16 W/16 D/16
        out = out.flatten(2).transpose(1, 2)  # ? N 128
        out = self.dblock1(out, H, W, D)  # ? N 128
        out = self.dnorm1(out)
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        out = F.relu6(F.interpolate(out, scale_factor=(2, 2, 2), mode='trilinear'))  # ? 128 H/16 W/16 D/16
        out = self.ba4(boundary_image=t4, guided_image=out)  # ? 128 H/16 W/16 D/16
        out,pred = self.sca_5(out)
        deep_preds = pred
        #######################################################
        # up 2
        B, C, H, W, D = out.shape
        out = self.dbn2(self.decoder2(out))  # ? 64 H/8 W/8 D/8
        out = out.flatten(2).transpose(1, 2)
        out = self.dblock2(out, H, W, D)
        out = self.dnorm2(out)
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        out = F.relu6(F.interpolate(out, scale_factor=(2, 2, 2), mode='trilinear'))
        out = self.ba3(boundary_image=t3, guided_image=out)  # ? 64 H/8 W/8 D/8
        out,pred = self.sca_4(out)
        deep_preds = deep_preds+ pred

        # up 3
        out = F.relu6(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        out = self.ba2(boundary_image=t2, guided_image=out)  # ? 32 H/4 W/4 D/4
        out,pred = self.sca_3(out)
        deep_preds = deep_preds+ pred

        out = F.relu6(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        out = self.ba1(boundary_image=t1, guided_image=out)  # ? 16 H/2 W/2 D/2
        out,pred = self.sca_2(out)
        deep_preds = deep_preds + pred
        #######################################################
        out = F.relu6(
            F.interpolate(self.dbn5(self.decoder5(out)), scale_factor=(2, 2, 2), mode='trilinear'))  # ? 4 H W D
        out,pred = self.sca_1(out)
        deep_preds = deep_preds + pred

        out = self.final(deep_preds)  # ? 2 H W D
        return out


if __name__ == "__main__":
    img = torch.ones([1, 1, 128, 128, 128])

    model = UNeXt3D_mlp_attention_boundary(n_channels=1, n_classes=2)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out_img = model(img)

    # print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]
    from thop import profile

    flops, params = profile(model, (img,))
    print('flops:', flops, 'params:', params)
    print('flops:%.2f G,params: %.2f M' % (flops / 1e9, params / 1e6))
