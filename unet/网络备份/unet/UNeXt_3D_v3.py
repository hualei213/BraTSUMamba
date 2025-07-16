# full assembly of the sub-parts to form the complete net
# H W D process same
import math

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from .unet_parts import *
# from unet_parts import *
from patchify import patchify
from timm.models.layers import DropPath

"""
    共享hierachical MLP
"""
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
        self.reshape_mlp_1 = nn.Linear(1024, 512)  # 这是个超参数，H*W*D*2,H*W*D需要根据当前输入图片大小进行调整
        self.reshape_mlp_2 = nn.Linear(128, 64)  # 这是个超参数，需要根据当前输入图片大小进行调整
        self.single_mlp_1 = nn.Linear(512, 512)  # 这是个超参数，需要根据当前输入图片大小进行调整
        self.single_mlp_2 = nn.Linear(64, 64)  # 这是个超参数，需要根据当前输入图片大小进行调整
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
        if H_size == 8:
            return self.reshape_mlp_1(x)
        if H_size == 4:
            return self.reshape_mlp_2(x)

    def choose_single_mlp(self, x, H_size):
        if H_size == 8:
            return self.single_mlp_1(x)
        if H_size == 4:
            return self.single_mlp_2(x)


    def hierachical_mlp(self, x, H, W, D, axis):
        count = 3  # 自定义移动多少次
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
        # x = self.filter(self.norm1(x),H,W,D)
        # H_W_D_Shift = self.norm2(self.mlp(self.norm2(x), H, W, D))
        H_W_D_Shift = self.norm2(self.mlp(x, H, W, D))
        x = x + self.drop_path(H_W_D_Shift)
        return x


class UNeXt3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNeXt3D, self).__init__()
        """
            使用hierachical mlp
            UNeXt3D V1
            3conv +token mlp
            buttom block 2
            token mlp + 3interpolate
        """
        # input converlution: ?x32x32x32x1 ==> ?x32x32x32x32
        self.dow_conv1 = inconv(n_channels, 16)
        self.dow_conv2 = inconv(16, 32)
        self.dow_conv3 = inconv(32, 64)

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

        self.norm1 = nn.LayerNorm(128)
        self.norm2 = nn.LayerNorm(256)
        self.norm3 = nn.LayerNorm(256)

        self.dnorm1 = nn.LayerNorm(128)

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

        out, H, W, D = self.patch_embed3(out)  # out:? N 256
        out = self.block3(out, H, W, D)  # out:? N 256
        out = self.norm3(out)
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()  # ? 256 H/32 W/32 D/32

        ### up stage
        # up1
        out = self.dbn1(self.decoder1(out))
        out = out.flatten(2).transpose(1, 2)  # ? N 128
        out = self.dblock1(out, H, W, D)  # ? N 128
        out = self.dnorm1(out)
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        out = F.relu6(F.interpolate(out, scale_factor=(2, 2, 2), mode='trilinear'))  # ? 128 H/16 W/16 D/16
        out = torch.add(out, t4)

        #######################################################
        # up 2
        out = F.relu6(
            F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2, 2), mode='trilinear'))  # ? 64 H/8 W/8 D/8
        out = torch.add(out, t3)
        # up 3
        out = F.relu6(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t2)  # ? 32 H/4 W/4 D/4
        out = F.relu6(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t1)  # ? 16 H/2 W/2 D/2
        #######################################################
        out = F.relu6(
            F.interpolate(self.dbn5(self.decoder5(out)), scale_factor=(2, 2, 2), mode='trilinear'))  # ? 4 H W D
        out = self.final(out)  # ? 2 H W D

        return out
