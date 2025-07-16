# full assembly of the sub-parts to form the complete net
# H W D process same
import math

import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.distributions.normal import Normal
from . import layers

from unet.unet_parts import *
# from unet_parts import *
from patchify import patchify
from timm.models.layers import DropPath

"""
在hierarchical MLP的基础上在网络的前3层增加MLP attention的操作
这里的MLP使用的是UNeXt的MLP结构
"""


#############新增加分支内容

class MLP_attention(nn.Module):
    def __init__(self, inchannle=3):
        super(MLP_attention, self).__init__()
        # self.cnn = nn.Conv3d(inchannle*2,inchannle,3,1,1)

    def forward(self, out_cnn, out_mlp):
        # b = F.softmax(out_mlp)
        a = F.softmax(out_mlp)
        residual_cnn = out_cnn
        out = out_cnn * a
        out = out + residual_cnn
        # out = torch.concat([out,residual_cnn],dim=1)
        # out = self.cnn(out)
        return out


class PWConv(nn.Module):
    def __init__(self, dim=128):
        super(PWConv, self).__init__()
        self.pointconv = nn.Conv3d(dim, dim, 1, 1, 0, bias=False)

    def forward(self, x, H, W, D):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W, D)
        x = self.pointconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class unext_shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.pwconv = PWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.fc3 = nn.Linear(hidden_features, out_features)

        self.shift_size = shift_size
        self.pad = shift_size // 2             #2

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
        ### shift domain
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)  # seplit xn along 1 axis
        x_shift = [torch.roll(x_c, shift, axis) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        ###tokenize domain
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_cat = torch.narrow(x_cat, 3, self.pad, W)
        x_s = torch.narrow(x_cat, 4, self.pad, D)
        ###tokens doamin
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
        x = self.dwconv(x, H, W, D)

        x = self.shift(x, H, W, D, 3)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.pwconv(x, H, W, D)

        x = self.shift(x, H, W, D, 4)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.act(x)  # GELU

        return x


class unext_shiftedBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=1., drop=0.2, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.drop_path = DropPath(drop) if drop > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = unext_shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
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

    def __init__(self, num_i, num_h, num_o):
        super(MLP, self).__init__()
        self.proj = OverlapPatchEmbed(patch_size=3, in_chans=num_i, embed_dim=num_o)
        self.block = unext_shiftedBlock(dim=num_o, mlp_ratio=1)
        self.norm = nn.LayerNorm(num_o)
        # self.cnn = nn.Conv3d(num_i, num_o, 3, 2, 1)

    def forward(self, x):
        B, _, _, _, _ = x.shape
        ## 输入影像首先做flatten
        out, H, W, D = self.proj(x)
        # residual = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        # MLP stage
        out = self.block(out, H, W, D)
        out = self.norm(out)
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        # 对MLP在加入一个残差
        # out = out + residual

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
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
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
        x = x.flatten(2).transpose(1, 2)  # (B,C,H*W*D)->(B,H*W*D,C)
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


class reshape_MLP(nn.Module):
    def __init__(self, input_size, out_size):
        super().__init__()
        self.proj = nn.Linear(input_size, out_size)

    def forward(self, x, H):
        return self.proj(x)


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
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def shift(self, x, H, W, D, axis, roll_rule):
        B, N, C = x.shape
        xn = x.transpose(1, 2).view(B, C, H, W, D).contiguous()
        ### shift domain
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)  # seplit xn along 1 axis
        x_shift = [torch.roll(x_c, shift, axis) for x_c, shift in zip(xs, roll_rule)]
        ###tokenize domain
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_cat = torch.narrow(x_cat, 3, self.pad, W)
        x_s = torch.narrow(x_cat, 4, self.pad, D)
        ###tokens doamin
        x_s = x_s.reshape(B, C, H * W * D).contiguous()
        # x = x_s.transpose(1, 2)
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
        count = 4  # 自定义移动多少次IBSR:4  NFBS:3
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
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.filter = Global_Filter(dim=dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
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


class UNeXt3D_mlp_attention(nn.Module):
    def __init__(self, n_channels, n_classes, bidir=False):
        super(UNeXt3D_mlp_attention, self).__init__()

        self.bidir = bidir

        # input converlution: ?x32x32x32x1 ==> ?x32x32x32x32

        self.dow_conv1_2 = inconv(8, 16)
        self.dow_conv1_1 = inconv(n_channels, 16)
        self.dow_conv2 = inconv(16, 32)
        self.dow_conv3 = inconv(32, 64)

        self.Bn_dow_conv1 = nn.BatchNorm3d(16)  # 批量归一化
        self.Bn_dow_conv2 = nn.BatchNorm3d(32)
        self.Bn_dow_conv3 = nn.BatchNorm3d(64)

        self.patch_embed1 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=64, embed_dim=128)
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=128, embed_dim=256)
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=1, in_chans=256, embed_dim=256)

        self.block1 = shiftedBlock(dim=128, mlp_ratio=1)
        self.block2 = shiftedBlock(dim=256, mlp_ratio=1)
        self.block3 = shiftedBlock(dim=256, mlp_ratio=1)

        self.dblock1_seg = shiftedBlock(dim=128, mlp_ratio=1)
        self.dblock1_reg = shiftedBlock(dim=128, mlp_ratio=1)

        self.norm1 = nn.LayerNorm(128)
        self.norm2 = nn.LayerNorm(256)
        self.norm3 = nn.LayerNorm(256)

        self.dnorm1_seg = nn.LayerNorm(128)
        self.dnorm1_reg = nn.LayerNorm(128)

        self.decoder1_seg = nn.Conv3d(256, 128, 3, stride=1, padding=1)
        self.decoder1_reg = nn.Conv3d(256, 128, 3, stride=1, padding=1)
        self.decoder2_seg = nn.Conv3d(128, 64, 3, stride=1, padding=1)
        self.decoder2_reg = nn.Conv3d(128, 64, 3, stride=1, padding=1)
        self.decoder3_seg = nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.decoder3_reg = nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.decoder4_seg = nn.Conv3d(32, 16, 3, stride=1, padding=1)
        self.decoder4_reg = nn.Conv3d(32, 16, 3, stride=1, padding=1)
        self.decoder5_seg = nn.Conv3d(16, 4, 3, stride=1, padding=1)
        self.decoder5_reg = nn.Conv3d(16, 4, 3, stride=1, padding=1)

        self.dbn1_seg = nn.BatchNorm3d(128)
        self.dbn2_seg = nn.BatchNorm3d(64)
        self.dbn3_seg = nn.BatchNorm3d(32)
        self.dbn4_seg = nn.BatchNorm3d(16)
        self.dbn5_seg = nn.BatchNorm3d(4)

        self.dbn1_reg = nn.BatchNorm3d(128)
        self.dbn2_reg = nn.BatchNorm3d(64)
        self.dbn3_reg = nn.BatchNorm3d(32)
        self.dbn4_reg = nn.BatchNorm3d(16)
        self.dbn5_reg = nn.BatchNorm3d(4)

        self.final_seg = nn.Conv3d(4, n_classes, kernel_size=1)
        self.final_reg = nn.Conv3d(4, n_classes, kernel_size=1)

        ###
        self.mlp1_1 = MLP(num_i=4, num_h=32, num_o=16)
        self.mlp1_2 = MLP(num_i=8, num_h=32, num_o=16)
        self.mlp2 = MLP(num_i=16, num_h=64, num_o=32)
        self.mlp3 = MLP(num_i=32, num_h=128, num_o=64)
        self.mlp_att1 = MLP_attention(16)
        self.mlp_att2 = MLP_attention(32)
        self.mlp_att3 = MLP_attention(64)

        ###

        # now we take care of any remaining convolutions
        final_convs = [32, 16, 16]
        ndims = 3
        prev_nf = 4
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):  # 32,16,16
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf
        prev_final_nf = prev_nf

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)  # Conv3d
        # self.flow = Conv(prev_final_nf, ndims, kernel_size=3, padding=1)
        self.flow = Conv(prev_final_nf, 3, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # configure optional resize layers (downsize)
        unet_half_res = False
        int_steps = 7
        int_downsize = 2
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure optional integration layer for diffeomorphic warp
        inshape = [128, 128, 128]
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)


    def segmentation_decoder(self,out,t1,t2,t3,t4,B,H,W,D):
        ### up stage
        # up1
        out = self.dbn1_seg(self.decoder1_seg(out))
        out = out.flatten(2).transpose(1, 2)  # ? N 128
        out = self.dblock1_seg(out, H, W, D)  # ? N 128
        out = self.dnorm1_seg(out)
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        out = F.relu6(F.interpolate(out, scale_factor=(2, 2, 2), mode='trilinear'))  # ? 128 H/16 W/16 D/16
        out = torch.add(out, t4)

        #######################################################
        # up 2
        out = F.relu6(
            F.interpolate(self.dbn2_seg(self.decoder2_seg(out)), scale_factor=(2, 2, 2), mode='trilinear'))  # ? 64 H/8 W/8 D/8
        out = torch.add(out, t3)
        # up 3
        out = F.relu6(F.interpolate(self.dbn3_seg(self.decoder3_seg(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t2)  # ? 32 H/4 W/4 D/4
        out = F.relu6(F.interpolate(self.dbn4_seg(self.decoder4_seg(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t1)  # ? 16 H/2 W/2 D/2
        #######################################################
        out = F.relu6(
            F.interpolate(self.dbn5_seg(self.decoder5_seg(out)), scale_factor=(2, 2, 2), mode='trilinear'))  # ? 4 H W D
        out = self.final_seg(out)  # ? 2 H W D
        return out

    def register(self,source,target,out,t1,t2,t3,t4,B,H,W,D):
        # up1
        out = self.dbn1_reg(self.decoder1_reg(out))
        out = out.flatten(2).transpose(1, 2)  # ? N 128
        out = self.dblock1_reg(out, H, W, D)  # ? N 128
        out = self.dnorm1_reg(out)
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        out = F.relu6(F.interpolate(out, scale_factor=(2, 2, 2), mode='trilinear'))  # ? 128 H/16 W/16 D/16
        out = torch.add(out, t4)
        # up 2
        out = F.relu6(
            F.interpolate(self.dbn2_reg(self.decoder2_reg(out)), scale_factor=(2, 2, 2), mode='trilinear'))  # ? 64 H/8 W/8 D/8
        out = torch.add(out, t3)
        # up 3
        out = F.relu6(F.interpolate(self.dbn3_reg(self.decoder3_reg(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t2)  # ? 32 H/4 W/4 D/4
        out = F.relu6(F.interpolate(self.dbn4_reg(self.decoder4_reg(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t1)  # ? 16 H/2 W/2 D/2
        #######################################################
        out = F.relu6(
            F.interpolate(self.dbn5_reg(self.decoder5_reg(out)), scale_factor=(2, 2, 2), mode='trilinear'))  # ? 4 H W D
        out = self.final_reg(out)  # ? 2 H W D
        x=out
        for conv in self.remaining:
            x = conv(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        source = source.float()
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        out = (y_source, preint_flow)

        # return non-integrated flow field if training
        # if not registration:
            # return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow)
            # out = (y_source, preint_flow)
        # else:
        #     out = (y_source, pos_flow)

        return out




    # def forward(self, source, target, flag, registration=False):
    def forward(self, source, target, flag):
        # target = torch.repeat_interleave(target, source.shape[0], dim=0)
        # if flag:
        #     x = torch.cat([source, target], dim=1)
        #     # x = x.float()
        #     mlp1 = self.mlp1_2(x)
        #     out = F.relu6(F.max_pool3d(self.Bn_dow_conv1(self.dow_conv1_2(x)), 2, 2))
        # else:
        #     x = source
        #     mlp1 = self.mlp1_1(x)
        #     out = F.relu6(F.max_pool3d(self.Bn_dow_conv1(self.dow_conv1_1(x)), 2, 2))
        # B = x.shape[0]  # ？ 1 H W D

        x_seg = source
        x_reg = torch.cat([source, target], dim=1)
        B = x_seg.shape[0]

        ##############新加入的MLP分支
        ### Tokenized MLP Stage
        mlp1_seg = self.mlp1_1(x_seg)
        mlp1_reg = self.mlp1_2(x_reg)
        mlp2_seg = self.mlp2(mlp1_seg)
        mlp2_reg = self.mlp2(mlp1_reg)
        mlp3_seg = self.mlp3(mlp2_seg)
        mlp3_reg = self.mlp3(mlp2_reg)
        ##############

        # input converlution stage
        out_seg = F.relu6(F.max_pool3d(self.Bn_dow_conv1(self.dow_conv1_1(x_seg)), 2, 2))
        out_reg = F.relu6(F.max_pool3d(self.Bn_dow_conv1(self.dow_conv1_2(x_reg)), 2, 2))
        out_seg = self.mlp_att1(out_seg, mlp1_seg)
        out_reg = self.mlp_att1(out_reg, mlp1_reg)
        t1_seg = out_seg  # ? 16 H/2 W/2 D/2
        t1_reg = out_reg
        out_seg = F.relu6(F.max_pool3d(self.Bn_dow_conv2(self.dow_conv2(out_seg)), 2, 2))
        out_reg = F.relu6(F.max_pool3d(self.Bn_dow_conv2(self.dow_conv2(out_reg)), 2, 2))
        out_seg = self.mlp_att2(out_seg, mlp2_seg)
        out_reg = self.mlp_att2(out_reg, mlp2_reg)
        t2_seg = out_seg  # ? 32 H/4 W/4 D/4
        t2_reg = out_reg
        out_seg = F.relu6(F.max_pool3d(self.Bn_dow_conv3(self.dow_conv3(out_seg)), 2, 2))
        out_reg = F.relu6(F.max_pool3d(self.Bn_dow_conv3(self.dow_conv3(out_reg)), 2, 2))
        out_seg = self.mlp_att3(out_seg, mlp3_seg)
        out_reg = self.mlp_att3(out_reg, mlp3_reg)
        t3_seg = out_seg  # ? 64 H/8 W/8 D/8
        t3_reg = out_reg
        ### Tokenized MLP Stage
        out_seg, H, W, D = self.patch_embed1(out_seg)  # ? N 128 N=H/16 w/16 D/16
        out_reg, H, W, D = self.patch_embed1(out_reg)
        out_seg = self.block1(out_seg, H, W, D)  # ? N 128
        out_reg = self.block1(out_reg, H, W, D)
        out_seg = self.norm1(out_seg)  # ? N 128
        out_reg = self.norm1(out_reg)
        out_seg = out_seg.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        out_reg = out_reg.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        t4_seg = out_seg  # ? 128 H/16 W/16 D/16
        t4_reg = out_reg
        # buttom block
        out_seg, H, W, D = self.patch_embed2(out_seg)  # ? N 256
        out_reg, H, W, D = self.patch_embed2(out_reg)
        out_seg = self.block2(out_seg, H, W, D)  # ? N 256
        out_reg = self.block2(out_reg, H, W, D)
        out_seg = self.norm2(out_seg)
        out_reg = self.norm2(out_reg)
        out_seg = out_seg.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()  # ? 256 H/32 W/32 D/32
        out_reg = out_reg.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()

        out_seg, H, W, D = self.patch_embed3(out_seg)  # out:? N 256
        out_reg, H, W, D = self.patch_embed3(out_reg)
        out_seg = self.block3(out_seg, H, W, D)  # out:? N 256
        out_reg = self.block3(out_reg, H, W, D)
        out_seg = self.norm3(out_seg)
        out_reg = self.norm3(out_reg)
        out_seg = out_seg.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()  # ? 256 H/32 W/32 D/32
        out_reg = out_reg.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()


        x_seg = self.segmentation_decoder(out_seg,t1_seg,t2_seg,t3_seg,t4_seg,B,H,W,D)
        if flag:
            x_reg = self.register(source,target,out_reg,t1_reg,t2_reg,t3_reg,t4_reg,B,H,W,D)
            return x_seg,x_reg
        else:
            return x_seg

        # return x_seg




class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


if __name__ == "__main__":
    img = torch.ones([1, 1, 128, 128, 128])

    model = UNeXt3D_mlp_attention(n_channels=1, n_classes=2)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out_img = model(img)

    print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]
    from thop import profile

    flops, params = profile(model, (img,))
    print('flops:', flops, 'params:', params)
    print('flops:%.2f G,params: %.2f M' % (flops / 1e9, params / 1e6))
