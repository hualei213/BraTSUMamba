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
        x = self.dwconv(x, H, W, D)
        x = self.act(x)  # GELU

        x = self.shift(x, H, W, D, 2)
        x = self.fc1(x)
        x = self.drop(x)

        x = self.shift(x, H, W, D, 3)
        x = self.fc2(x)
        x = self.drop(x)

        x = self.shift(x, H, W, D, 4)
        x = self.fc3(x)
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

class tokenization(nn.Module):
    def __init__(self, input_channel_dim=64, hidden_dim=384):
        super().__init__()

        self.mlp1 = nn.Linear(in_features=input_channel_dim, out_features=hidden_dim)
        self.mlp2 = nn.Linear(in_features=input_channel_dim, out_features=512)

    def forward(self, x):
        """
        :param x: shape ?xNxC
        :return:
        """
        out1 = self.mlp1(x)
        out1 = F.softmax(out1, dim=1).transpose(2, 1)
        out2 = self.mlp2(x)
        out = torch.matmul(out1, out2)
        return out


class detokenization(nn.Module):
    def __init__(self, input_channel_dim=64, hidden_dim=384):
        super().__init__()
        self.mlp1 = nn.Linear(in_features=input_channel_dim, out_features=hidden_dim, bias=False)
        self.mlp2 = nn.Linear(in_features=512, out_features=512)
        self.mlp3 = nn.Linear(in_features=input_channel_dim, out_features=hidden_dim)

    def forward(self, Xc, T_prev):
        """
        :param x: shape ?xNxC
        :return:
        """
        x_out = self.mlp1(Xc)
        out = self.mlp2(torch.matmul(x_out, T_prev)).transpose(2, 1)
        out = F.softmax(out, dim=1)
        out = torch.matmul(out, self.mlp3(Xc))
        out = out.transpose(2, 1)
        return out

class MixerBlock(nn.Module):
    def __init__(self, input_dim, hidden_token_dim):
        """
        :param input_dim: token input dim =256
        :param hidden_dim: dim = 384
        :param channel_dim:
        :param dropout:
        """
        super().__init__()
        self.mlp1_1 = nn.Linear(in_features=hidden_token_dim, out_features=hidden_token_dim)
        self.gelu1 = nn.GELU()
        self.mlp1_2 = nn.Linear(in_features=hidden_token_dim, out_features=hidden_token_dim)
        self.LayerNorm1 = nn.LayerNorm(hidden_token_dim)
        self.mlp2_1 = nn.Linear(in_features=hidden_token_dim, out_features=768)  # 768对应D_c
        self.gelu2 = nn.GELU()
        self.mlp2_2 = nn.Linear(in_features=768, out_features=hidden_token_dim)
        self.LayerNorm2 = nn.LayerNorm(512)

    def forward(self, T_in):
        # T_in B N C
        # token_mixer
        out = T_in.transpose(2, 1)
        out = self.mlp1_1(out)
        out = self.gelu1(out)
        out = self.mlp1_2(out)
        out = self.LayerNorm1(out)  # B C N
        out = out.transpose(2, 1)
        U = T_in + out  # 公式3
        # channel mixer
        out = U.transpose(2, 1)
        out = self.mlp2_1(out)
        out = self.gelu2(out)
        out = self.mlp2_2(out)
        out = out.transpose(2, 1)
        out = self.LayerNorm2(out)
        out = T_in + out
        return out
class toeken_proj(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=383):
        super().__init__()
        self.linear1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.linear2 = nn.Linear(in_features=512, out_features=hidden_dim)
        self.linear3 = nn.Linear(in_features=512, out_features=input_dim)

    def forward(self, X_c, Tout):
        X_c_out = self.linear1(X_c)
        out = self.linear2(Tout).transpose(2, 1)
        out = F.softmax(torch.matmul(X_c_out, out), dim=1)
        liner3 = self.linear3(Tout)
        out = X_c + torch.matmul(out, liner3)

        return out

class down_MLP_Block(nn.Module):
    def __init__(self, in_channel=32, out_channel=64, tokenize_number=384):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=2,
                              bias=True)
        self.tokenizer = tokenization(input_channel_dim=out_channel, hidden_dim=tokenize_number)
        self.block1 = nn.Sequential(*[
            MixerBlock(input_dim=out_channel, hidden_token_dim=tokenize_number) for _ in range(4)
        ]
                                    )
        self.proj = toeken_proj(input_dim=out_channel, hidden_dim=tokenize_number)
        self.drop = DropPath(0.2)

    def overlapPatchEmbed(self, x):
        x = self.conv(x)
        B, C, H, W, D = x.shape
        x = x.flatten(2).transpose(2, 1)
        return x, H, W, D

    def forward(self, x):
        B = x.shape[0]
        # # early conv + flatten
        xc, H, W, D = self.overlapPatchEmbed(x)
        # tokenization output =  T_in
        out = self.tokenizer(xc)

        T_in = out
        # MLP stage
        for blk in self.block1:
            out = blk(out)

        X_out = self.proj(xc, out)
        # X_out = self.drop(X_out)
        X_out = X_out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        return X_out, T_in


class up_MLP_Block(nn.Module):
    def __init__(self, in_channel=32, out_channel=64, tokenize_number=384):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_channels=in_channel, out_channels=out_channel, kernel_size=2, padding=0,
                                       stride=2,
                                       bias=True)
        self.tokenizer = detokenization(input_channel_dim=out_channel, hidden_dim=tokenize_number)
        self.block1 = nn.Sequential(
            *[MixerBlock(input_dim=out_channel, hidden_token_dim=tokenize_number) for _ in range(4)]
        )
        self.proj = toeken_proj(input_dim=out_channel, hidden_dim=tokenize_number)

    def overlapPatchEmbed(self, x):
        x = self.conv(x)
        B, C, H, W, D = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, H, W, D

    def forward(self, x, T_prew):
        B = x.shape[0]
        # # early conv + flatten
        xc, H, W, D = self.overlapPatchEmbed(x)
        # tokenization
        out = self.tokenizer(xc, T_prew)
        for blk in self.block1:
            out = blk(out)

        X_out = self.proj(xc, out)
        X_out = X_out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()

        return X_out


class down_residual_conv_block(nn.Module):
    def __init__(self, in_channels=16, out_channels=32):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=2,
                               bias=True)
        self.conv1_1 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1,
                                 stride=1,
                                 bias=True)
        self.conv2 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0,
                               stride=2, bias=True)

    def forward(self, x):
        # (3,3,3)->(3,3,3)
        out1 = self.conv1(x)
        out1 = self.conv1_1(out1)
        # (1,1,1)
        out2 = self.conv2(x)
        out = torch.add(out1, out2)
        return out


class up_residual_conv_block(nn.Module):
    def __init__(self, in_channels=16, out_channels=32):
        super().__init__()
        self.conv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, padding=0,
                                        stride=2, bias=True)
        self.conv1_1 = nn.ConvTranspose3d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0,
                                          stride=1, bias=True)
        self.conv2 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, padding=0,
                                        stride=2, bias=True)

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.conv1_1(out1)
        out2 = self.conv2(x)
        out = torch.add(out1, out2)
        return out

class UNeXt_MLP_VNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNeXt_MLP_VNet, self).__init__()
        self.dow_conv1 = inconv(n_channels, 16)
        self.dow_conv2 = inconv(16, 32)

        self.Bn_dow_conv1 = nn.BatchNorm3d(16)
        self.Bn_dow_conv2 = nn.BatchNorm3d(32)

        self.dow_mlp_block1 = down_MLP_Block(in_channel=32, out_channel=64, tokenize_number=384)
        self.dow_mlp_block2 = down_MLP_Block(in_channel=64, out_channel=128, tokenize_number=196)
        self.dow_mlp_block3 = down_MLP_Block(in_channel=128, out_channel=256, tokenize_number=98)

        self.up_mlp_block1 = up_MLP_Block(in_channel=256, out_channel=128, tokenize_number=196)
        self.up_mlp_block2 = up_MLP_Block(in_channel=128, out_channel=64, tokenize_number=384)


        self.decoder3 = nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv3d(32, 16, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv3d(16, 4, 3, stride=1, padding=1)

        self.dbn3 = nn.BatchNorm3d(32)
        self.dbn4 = nn.BatchNorm3d(16)
        self.dbn5 = nn.BatchNorm3d(4)

        self.final = nn.Conv3d(4, n_classes, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]
        out = F.relu6(F.max_pool3d(self.Bn_dow_conv1(self.dow_conv1(x)), 2, 2))
        t1 = out
        out = F.relu6(F.max_pool3d(self.Bn_dow_conv2(self.dow_conv2(out)), 2, 2))
        t2 = out
        # mlp mixer stage
        x3_out, T3_in = self.dow_mlp_block1(out)
        x4_out, T4_in = self.dow_mlp_block2(x3_out)
        t4 = x4_out
        x5_out, T5_in = self.dow_mlp_block3(x4_out)
        t5 = x5_out
        ### up stage
        ## deconder stage
        out = self.up_mlp_block1(x5_out, T4_in)
        out = self.up_mlp_block2(out, T3_in)

        #######################################################
        # up 3
        out = F.relu6(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t2)
        out = F.relu6(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t1)
        out = F.relu6(F.interpolate(self.dbn5(self.decoder5(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        out = self.final(out)

        return out


if __name__ == "__main__":
    """
    12312
    """
    device = torch.device("cpu")
    input_data = torch.randn((1, 1, 128, 128, 128))
    model = UNeXt_MLP_VNet(1, 2)
    input_data = input_data.to(device)
    model.to(device)
    output_data = model(input_data)
    print(output_data.shape)
