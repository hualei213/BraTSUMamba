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
# from timm.models.layers import DropPath



class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=2,in_chans=3, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=2, padding=1,bias=False)

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
    def forward(self, x,H, W, D):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W, D)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
class PWConv(nn.Module):
    def __init__(self, dim=128):
        super(PWConv, self).__init__()
        self.pointconv = nn.Conv3d(dim,dim,1,1,0,bias=False)
    def forward(self, x,H, W, D):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W, D)
        x = self.pointconv(x)
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
        self.pwconv=PWConv(hidden_features)
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
        elif isinstance(m, nn.Conv2d):
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


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class shiftedBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=1., drop=0.2, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.drop_path = DropPath(drop) if drop > 0. else nn.Identity()
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
        H_W_D_Shift = self.mlp(self.norm2(x), H, W, D)
        x = x + self.drop_path(H_W_D_Shift)
        return x


class unet_mlp(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(unet_mlp, self).__init__()

        # input converlution: ?x32x32x32x1 ==> ?x32x32x32x32
        self.dow_conv1 = inconv(n_channels, 16)
        self.dow_conv2 = inconv(16, 32)
        self.dow_conv3 = inconv(32, 64)

        self.mlp1 = nn.Linear(1, 16)
        self.mlp2 = nn.Linear(16, 32)
        self.mlp3 = nn.Linear(32, 64)
        self.patch_embed1 = OverlapPatchEmbed(patch_size=3, in_chans=16, embed_dim=16)
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, in_chans=32, embed_dim=32)
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, in_chans=64, embed_dim=64)


        self.Bn_dow_conv1 = nn.BatchNorm3d(16)
        self.Bn_dow_conv2 = nn.BatchNorm3d(32)
        self.Bn_dow_conv3 = nn.BatchNorm3d(64)

        self.patch_embed4 = OverlapPatchEmbed(patch_size=3,  in_chans=64, embed_dim=128)
        self.patch_embed5 = OverlapPatchEmbed(patch_size=3,  in_chans=128, embed_dim=256)

        self.block1 = shiftedBlock(dim=128, mlp_ratio=1)
        self.block2 = shiftedBlock(dim=256, mlp_ratio=1)

        self.dblock1 = shiftedBlock(dim=128, mlp_ratio=1)
        self.dblock2 = shiftedBlock(dim=64, mlp_ratio=1)

        self.norm1 = nn.LayerNorm(128)
        self.norm2 = nn.LayerNorm(256)

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

    def forward(self, x):
        B,C,H,W,D = x.shape
        # input converlution stage
        out = F.relu6(F.max_pool3d(self.Bn_dow_conv1(self.dow_conv1(x)), 2, 2))
        t1 = out
        out = F.relu6(F.max_pool3d(self.Bn_dow_conv2(self.dow_conv2(out)), 2, 2))
        t2 = out
        out = F.relu6(F.max_pool3d(self.Bn_dow_conv3(self.dow_conv3(out)), 2, 2))
        t3 = out
        """
        out = x.flatten(2).transpose(1, 2)
        out = self.mlp1(out).reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        out = F.max_pool3d(out, 2, 2)
        t1 =out
        # out,H,W,D = self.patch_embed1(out)
        # t1 = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()

        H,W,D = H//2,W//2,D//2
        out = out.flatten(2).transpose(1, 2)
        out = self.mlp2(out).reshape(B, H , W , D , -1).permute(0, 4, 1, 2, 3).contiguous()
        out = F.max_pool3d(out, 2, 2)
        t2=out
        # out, H, W, D = self.patch_embed2(out)
        # t2 = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()

        H, W, D = H // 2, W // 2, D // 2
        out = out.flatten(2).transpose(1, 2)
        out = self.mlp3(out).reshape(B, H , W , D , -1).permute(0, 4, 1, 2, 3).contiguous()
        out = F.max_pool3d(out, 2, 2)
        # out,H,W,D = self.patch_embed3(out)
        # out=out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        t3 = out
"""
        ### Tokenized MLP Stage
        ### Token Mlp domain 1
        out, H, W, D = self.patch_embed4(out)
        out = self.block1(out, H, W, D)
        out = self.norm1(out)
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        t4 = out
        # Token Mlp domain 2
        out, H, W, D = self.patch_embed5(out)  # out:(64,9,3,128) B,H,D*W,C
        out = self.block2(out, H, W, D)
        out = self.norm2(out)
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()

        ### up stage
        # up1
        out = F.relu6(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t4)
        _, _, H, W, D = out.shape
        out = out.flatten(2).transpose(1, 2)
        out = self.dblock1(out, H, W, D)
        out = self.dnorm1(out)
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        # up 2
        out = F.relu6(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t3)

        _, _, H, W, D = out.shape
        out = out.flatten(2).transpose(1, 2)
        out = self.dblock2(out, H, W, D)
        out = self.dnorm2(out)
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()

        #######################################################
        # up 3
        out = F.relu6(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t2)
        out = F.relu6(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t1)
        out = F.relu6(F.interpolate(self.dbn5(self.decoder5(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        out = self.final(out)

        return out
