# full assembly of the sub-parts to form the complete net
# H W D process same
import math

import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.distributions.normal import Normal
# from . import layers
from unet.unet_parts import *
# from unet_parts import *
from patchify import patchify
from timm.models.layers import DropPath
from unet.Mamba_my_4x_fusion import Mamba
from unet.MDAF import MDAF

"""
"""
def gaussian_blur_3d(image, kernel_size=5, sigma=1.0):
    x = torch.arange(kernel_size).float() - kernel_size // 2
    gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel_1d = gauss / gauss.sum()
    kernel_3d = kernel_1d.view(1, 1, -1, 1, 1) * kernel_1d.view(1, 1, 1, -1, 1) * kernel_1d.view(1, 1, 1, 1, -1)
    kernel_3d = kernel_3d.to(image.device)

    # 分离通道进行 3D 卷积
    channels = image.shape[1]
    return F.conv3d(image, kernel_3d.repeat(channels, 1, 1, 1, 1), padding=kernel_size // 2, groups=channels)

# 拉普拉斯金字塔分解（3D 版）
def laplacian_pyramid_3d(image, levels=3):
    current = image
    pyramid = []
    for _ in range(levels):
        blurred = gaussian_blur_3d(current, sigma=1.0)
        downsampled = F.avg_pool3d(blurred, kernel_size=2)
        pyramid.append(current - F.interpolate(downsampled, size=current.shape[-3:], mode='trilinear'))
        current = downsampled
    low_freq = F.interpolate(current, size=image.shape[-3:], mode='trilinear')
    high_freq = image - low_freq
    return high_freq

class FeatureFusion3D(nn.Module):
    def __init__(self, channels):
        """
        Feature Fusion module for 3D inputs (B, C, H, W, D).

        Args:
            channels: Number of input channels (C).
        """
        super(FeatureFusion3D, self).__init__()

        # Global Pooling (averaging over H, W, D)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # First branch (global features)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=1)

        # Second branch (spatial features)
        self.conv3 = nn.Conv3d(channels, channels, kernel_size=1)
        self.relu2 = nn.ReLU()
        self.conv4 = nn.Conv3d(channels, channels, kernel_size=1)

        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass for 3D Feature Fusion.

        Args:
            x: Input tensor of shape (B, C, H, W, D).
        Returns:
            Output tensor of shape (B, C, H, W, D).
        """
        # Global pooling branch
        global_pooled = self.global_pool(x)  # Shape: (B, C, 1, 1, 1)
        global_features = self.conv1(global_pooled)  # Shape: (B, C, 1, 1, 1)
        global_features = self.relu1(global_features)
        global_features = self.conv2(global_features)  # Shape: (B, C, 1, 1, 1)

        # Spatial feature branch
        spatial_features = self.conv3(x)  # Shape: (B, C, H, W, D)
        spatial_features = self.relu2(spatial_features)
        spatial_features = self.conv4(spatial_features)  # Shape: (B, C, H, W, D)


        # Element-wise addition
        combined_features = global_features + spatial_features  # Broadcasting for (B, C, H, W, D)

        # Sigmoid activation
        attention_weights = self.sigmoid(combined_features)  # Shape: (B, C, H, W, D)



        return attention_weights

        # Apply attention weights
        # output = x * attention_weights  # Element-wise multiplication

#MlpChannel
class MlpChannel(nn.Module):
    def __init__(self,hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#MambaLayer
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices_small=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type="v3",
            nslices_small=num_slices_small,
        )
        self.fusion = FeatureFusion3D(channels=dim)

    def forward(self, x):
        B, C = x.shape[:2]
        x_skip = x.cuda()
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_flat = x_flat.cuda()
        x_norm = self.norm(x_flat)
        x_sq,x_slice = self.mamba(x_norm)

        # out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out_sq = x_sq.transpose(-1, -2).reshape(B, C, *img_dims)
        out_slice = x_slice.transpose(-1, -2).reshape(B, C, *img_dims)
        out = out_sq+out_slice
        wff = self.fusion(out)
        out = out_sq * wff + (1-wff) * out_slice
        out = out + x_skip

        return out



#GSC
class GSC(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        self.nonliner2 = nn.ReLU()

        self.proj7 = nn.Conv3d(in_channles, in_channles, 7, 2, 3)
        self.norm7 = nn.InstanceNorm3d(in_channles)
        self.nonliner7 = nn.ReLU()


        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):
        x_residual = x

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        # x1 = self.proj7(x1)
        # x1 = self.norm7(x1)
        # x1 = self.nonliner7(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        # x = x1 + x2
        x = x1 * x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)

        return x + x_residual


#Mamba
class MambaEncoder(nn.Module):
    def __init__(self, in_chans=4, depths=[2, 2, 2, 2], dims=[16, 32, 64, 128],drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
              nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
              )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        num_slices_small_list = [64, 32, 16, 8]
        cur = 0
        for i in range(4):
            gsc = GSC(dims[i])

            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i], num_slices_small=num_slices_small_list[i]) for j in range(depths[i])]
            )

            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            x = self.stages[i](x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x





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


class gmsm_4x_pc_hlcrossatt(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(gmsm_4x_pc_hlcrossatt, self).__init__()

        in_chans = 4
        depths = [2, 2, 2, 2]
        feat_size = [16, 32, 64, 128]
        drop_path_rate = 0
        layer_scale_init_value = 1e-6
        self.vit = MambaEncoder(in_chans,
                                depths=depths,
                                dims=feat_size,
                                drop_path_rate=drop_path_rate,
                                layer_scale_init_value=layer_scale_init_value
                                )

        # input converlution: ?x32x32x32x1 ==> ?x32x32x32x32

        self.dow_conv1 = inconv(n_channels, 16)
        self.dow_conv2 = inconv(16, 32)
        self.dow_conv3 = inconv(32, 64)

        self.Bn_dow_conv1 = nn.BatchNorm3d(16)  # 批量归一化
        self.Bn_dow_conv2 = nn.BatchNorm3d(32)
        self.Bn_dow_conv3 = nn.BatchNorm3d(64)

        # self.patch_embed1 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=64, embed_dim=128)
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
        self.decoder5 = nn.Conv3d(16, 8, 3, stride=1, padding=1)  #out_channels 4-->8

        self.dbn1 = nn.BatchNorm3d(128)
        self.dbn2 = nn.BatchNorm3d(64)
        self.dbn3 = nn.BatchNorm3d(32)
        self.dbn4 = nn.BatchNorm3d(16)
        self.dbn5 = nn.BatchNorm3d(8)#4-->8


        self.final = nn.Conv3d(8, n_classes, kernel_size=1)#4-->8

        ###
        # self.mlp1 = MLP(num_i=4, num_h=32, num_o=16)
        # self.mlp2 = MLP(num_i=16, num_h=64, num_o=32)
        # self.mlp3 = MLP(num_i=32, num_h=128, num_o=64)
        # self.mlp_att1 = MLP_attention(16)
        # self.mlp_att2 = MLP_attention(32)
        # self.mlp_att3 = MLP_attention(64)


        self.conv_4 = nn.Conv3d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.conv_3 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv_2 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv_1 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0)

        self.conv_fre111_4 = nn.Conv3d(128, 128, kernel_size=1)
        self.conv_fre111_3 = nn.Conv3d(64, 64, kernel_size=1)
        self.conv_fre111_2 = nn.Conv3d(32, 32, kernel_size=1)
        self.conv_fre111_1 = nn.Conv3d(16, 16, kernel_size=1)


        self.MDAF4 = MDAF(128,8,'WithBias')
        self.MDAF3 = MDAF(64,8,'WithBias')
        self.MDAF2 = MDAF(32,8,'WithBias')
        self.MDAF1 = MDAF(16,8,'WithBias')







    def forward_encoder(self,x):

        mamba_outs = self.vit(x)

        B = x.shape[0]
        ##############加入的MLP分支
        ### Tokenized MLP Stage
        # mlp1 = self.mlp1(x)
        # mlp2 = self.mlp2(mlp1)
        # mlp3 = self.mlp3(mlp2)
        ##############


        # input converlution stage
        # out = F.relu6(F.max_pool3d(self.Bn_dow_conv1(self.dow_conv1(x)), 2, 2))
        # out = self.mlp_att1(out, mlp1)
        # t1 = out                     #encoder_1
        out = mamba_outs[0]
        t1 = out
        # out = F.relu6(F.max_pool3d(self.Bn_dow_conv2(self.dow_conv2(out)), 2, 2))
        # out = self.mlp_att2(out, mlp2)
        # t2 = out  # ? 32 H/4 W/4 D/4       encoder_2
        out = mamba_outs[1]
        t2 = out
        # out = F.relu6(F.max_pool3d(self.Bn_dow_conv3(self.dow_conv3(out)), 2, 2))
        # out = self.mlp_att3(out, mlp3)
        # t3 = out  # ? 64 H/8 W/8 D/8       encoder_3
        out = mamba_outs[2]
        t3 = out
        ### Tokenized MLP Stage
        # out, H, W, D = self.patch_embed1(out)  # ? N 128 N=H/16 w/16 D/16
        # out = self.block1(out, H, W, D)  # ? N 128
        # out = self.norm1(out)  # ? N 128
        # out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        # t4 = out  # ? 128 H/16 W/16 D/16    encoder_4
        out = mamba_outs[3]
        t4 = out
        # buttom block
        out, H, W, D = self.patch_embed2(out)  # ? N 256
        out = self.block2(out, H, W, D)  # ? N 256
        out = self.norm2(out)
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()  # ? 256 H/32 W/32 D/32                 encoder5
        # out = mamba_outs[4]
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
        out = self.dblock1(out, 4, 4, 4)  # ? N 128
        out = self.dnorm1(out)
        out = out.reshape(B, 4, 4, 4, -1).permute(0, 4, 1, 2, 3).contiguous()
        out = F.relu6(F.interpolate(out, scale_factor=(2, 2, 2), mode='trilinear'))  # ? 128 H/16 W/16 D/16
        # out = torch.add(out, t4)
        out = torch.cat([out,t4],dim=1)
        out = self.conv_4(out)
        high_f4 = laplacian_pyramid_3d(out,3)
        high_f4 = self.conv_fre111_4(high_f4)
        out = out + self.MDAF4(out,high_f4)+self.MDAF4(out,out)
        #######################################################
        # up 2
        out = F.relu6(
            F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2, 2), mode='trilinear'))  # ? 64 H/8 W/8 D/8
        out = torch.cat([out, t3], dim=1)
        out = self.conv_3(out)

        high_f3 = laplacian_pyramid_3d(out,3)
        high_f3 = self.conv_fre111_3(high_f3)
        out = out + self.MDAF3(out, high_f3) + self.MDAF3(out, out)

        # up 3
        out = F.relu6(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.cat([out, t2], dim=1)
        out = self.conv_2(out)

        high_f2 = laplacian_pyramid_3d(out,3)
        high_f2 = self.conv_fre111_2(high_f2)
        out = out + self.MDAF2(out, high_f2) + self.MDAF2(out, out)

        #up 4
        out = F.relu6(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.cat([out, t1], dim=1)
        out = self.conv_1(out)

        high_f1 = laplacian_pyramid_3d(out, 3)
        high_f1 = self.conv_fre111_1(high_f1)
        out = out + self.MDAF1(out, high_f1) + self.MDAF1(out, out)

        #up 5
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
    img = torch.ones([1, 4, 128, 128, 128]).cuda()

    model = gmsm_4x_pc_hlcrossatt(n_channels=4, n_classes=4)
    model = model.cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out_img = model(img)

    print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]
    from thop import profile

    flops, params = profile(model, (img,))
    print('flops:', flops, 'params:', params)
    print('flops:%.2f G,params: %.2f M' % (flops / 1e9, params / 1e6))
