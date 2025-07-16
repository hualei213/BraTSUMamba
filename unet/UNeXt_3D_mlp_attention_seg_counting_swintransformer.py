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
from unet.SwinTransformer import  SwinTransformerSys3D

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
    def __init__(self, n_channels, n_classes,embed_dim=24, win_size=7):
        super(UNeXt3D_mlp_attention, self).__init__()


        self.embed_dim = embed_dim
        self.win_size = win_size
        self.win_size = (self.win_size, self.win_size, self.win_size)
        self.fusion_Conv3D_1 = nn.Conv3d(in_channels=
                                       48,out_channels=24,kernel_size=3,padding=1,stride=1)
        self.fusion_Conv3D_2 = nn.Conv3d(in_channels=
                                         96, out_channels=48, kernel_size=3, padding=1, stride=1)
        self.fusion_Conv3D_3 = nn.Conv3d(in_channels=
                                         192, out_channels=96, kernel_size=3, padding=1, stride=1)


        self.swin_transformer = SwinTransformerSys3D(img_size=(128, 128, 128),
                                              patch_size=(2, 2, 2),
                                              in_chans=4,
                                              embed_dim=self.embed_dim,
                                              depths=[2, 2, 2],
                                              num_heads=[3, 6, 12],
                                              window_size=self.win_size,
                                              mlp_ratio=4.,
                                              qkv_bias=True,
                                              qk_scale=None,
                                              drop_rate=0.,
                                              attn_drop_rate=0.,
                                              drop_path_rate=0.1,
                                              norm_layer=nn.LayerNorm,
                                              patch_norm=True,
                                              use_checkpoint=False,
                                              frozen_stages=-1)



        # input converlution: ?x32x32x32x1 ==> ?x32x32x32x32

        self.dow_conv1 = inconv(n_channels, 24)
        self.dow_conv2 = inconv(24, 48)
        self.dow_conv3 = inconv(48, 96)

        self.Bn_dow_conv1 = nn.BatchNorm3d(24)  # 批量归一化
        self.Bn_dow_conv2 = nn.BatchNorm3d(48)
        self.Bn_dow_conv3 = nn.BatchNorm3d(96)

        self.patch_embed1 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=96, embed_dim=192)
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=192, embed_dim=384)
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=1, in_chans=384, embed_dim=384)

        self.block1 = shiftedBlock(dim=192, mlp_ratio=1)
        self.block2 = shiftedBlock(dim=384, mlp_ratio=1)
        self.block3 = shiftedBlock(dim=384, mlp_ratio=1)

        self.dblock1 = shiftedBlock(dim=192, mlp_ratio=1)

        self.norm1 = nn.LayerNorm(192)
        self.norm2 = nn.LayerNorm(384)
        self.norm3 = nn.LayerNorm(384)

        self.dnorm1 = nn.LayerNorm(192)

        self.decoder1 = nn.Conv3d(384, 192, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv3d(192, 96, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv3d(96, 48, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv3d(48, 24, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv3d(24, 8, 3, stride=1, padding=1)  #out_channels 4-->8

        self.dbn1 = nn.BatchNorm3d(192)
        self.dbn2 = nn.BatchNorm3d(96)
        self.dbn3 = nn.BatchNorm3d(48)
        self.dbn4 = nn.BatchNorm3d(24)
        self.dbn5 = nn.BatchNorm3d(8)#4-->8


        self.final = nn.Conv3d(8, n_classes, kernel_size=1)#4-->8

        ###
        self.mlp1 = MLP(num_i=4, num_h=32, num_o=24)
        self.mlp2 = MLP(num_i=24, num_h=64, num_o=48)
        self.mlp3 = MLP(num_i=48, num_h=128, num_o=96)
        self.mlp_att1 = MLP_attention(24)
        self.mlp_att2 = MLP_attention(48)
        self.mlp_att3 = MLP_attention(96)

    def forward_encoder(self,x):
        B = x.shape[0]
        ##############加入的MLP分支
        ### Tokenized MLP Stage
        mlp1 = self.mlp1(x)
        mlp2 = self.mlp2(mlp1)
        mlp3 = self.mlp3(mlp2)
        ##############


        #swin-transformer分支
        x_transformer, x_downsample, v_values_1, k_values_1, q_values_1, v_values_2, k_values_2, q_values_2 = self.swin_transformer(x)


        # input converlution stage
        out = F.relu6(F.max_pool3d(self.Bn_dow_conv1(self.dow_conv1(x)), 2, 2))
        out = self.mlp_att1(out, mlp1)
        out = torch.cat((out, x_downsample[0]), dim=1)
        out = self.fusion_Conv3D_1(out)
        t1 = out  # ? 16 H/2 W/2 D/2       encoder_1
        out = F.relu6(F.max_pool3d(self.Bn_dow_conv2(self.dow_conv2(out)), 2, 2))
        out = self.mlp_att2(out, mlp2)
        out = torch.cat((out, x_downsample[1]), dim=1)
        out = self.fusion_Conv3D_2(out)
        t2 = out  # ? 32 H/4 W/4 D/4       encoder_2
        out = F.relu6(F.max_pool3d(self.Bn_dow_conv3(self.dow_conv3(out)), 2, 2))
        out = self.mlp_att3(out, mlp3)
        out = torch.cat((out, x_downsample[2]), dim=1)
        out = self.fusion_Conv3D_3(out)
        t3 = out  # ? 64 H/8 W/8 D/8       encoder_3
        ### Tokenized MLP Stage
        out, H, W, D = self.patch_embed1(out)  # ? N 128 N=H/16 w/16 D/16
        out = self.block1(out, H, W, D)  # ? N 128
        out = self.norm1(out)  # ? N 128
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        t4 = out  # ? 128 H/16 W/16 D/16    encoder_4
        # buttom block
        out, H, W, D = self.patch_embed2(out)  # ? N 256
        out = self.block2(out, H, W, D)  # ? N 256
        out = self.norm2(out)
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()  # ? 256 H/32 W/32 D/32                 encoder5
        return t1,t2,t3,t4,out


    def forward_decoder(self,out,t1,t2,t3,t4):

        B = out.shape[0]
        out, H, W, D = self.patch_embed3(out)  # out:? N 256
        out = self.block3(out, H, W, D)  # out:? N 256
        out = self.norm3(out)
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()  # ? 256 H/32 W/32 D/32


        # decoder
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


    def forward(self, source):
        t1,t2,t3,t4,out=self.forward_encoder(source)
        out = self.forward_decoder(out,t1,t2,t3,t4)
        x = out
        return x



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
