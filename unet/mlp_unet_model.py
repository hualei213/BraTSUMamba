# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import torch
import torch.nn as nn


from .unet_parts import *
# from unet_parts import *
# from patchify import patchify
class shift_Windows(nn.Module):
    def __init__(self, ty=None, S=32, C=32,patch_size=2):
        super(shift_Windows, self).__init__()
        self.ty = ty
        self.S = S
        self.C = C
        self.patch_size = patch_size
        if ty == "shift_token":
            self.linear = nn.Linear(S * C, C)
        else:
            self.linear = nn.Linear(S * C, S * C)
    def shift(self,x,strid):
        # input x (B,H,W,D,T,C)
        # shift roll T dim
        return torch.roll(x,shifts=(strid),dims=(4))
    '''
        input x shape (B,C,T,H,W,D)
        need x shape  (B,H,W,T,C)
    '''
    def group_time_mixing(self, x):
        B, C,H, W, D = x.shape
        # get patches
        x = x.reshape(B, C, -1, H // self.patch_size, W // self.patch_size, D // self.patch_size)
        x = x.permute(0, 3, 4, 5, 2, 1).contiguous()
        B,H,W,D,T,C = x.shape
        if self.ty == "short_range":
            x = self.linear(x.reshape(B,H,W,D, -1, self.S * C))
            x = x.reshape(B,H,W,D,T,C)
        elif self.ty == "long_range":
            x = x.reshape(B,H,W,D,self.S,-1,C).transpose(4,5)
            x = self.linear(x.reshape(B,H,W,D,-1,self.S*C))
            x = x.reshape(B,H,W,D,-1,self.S,C).transpose(4,5)
            x = x.reshape(B,H,W,D,T,C)
        elif self.ty == "shift_window":
            x = self.shift(x,self.S//2)
            x=self.linear(x.reshape(B,H,W,D,-1,self.S*C))
            x= self.shift(x.reshape(B,H,W,D,T,C),-self.S//2)
        elif self.ty == "shift_token":
            x = [self.shift(x,i) for i in range(self.S)]
            x = self.linear(torch.cat(x, dim=5))

        # tansform dims shape from (B,H,W,D,T,C) to (B,C,T,H,W,D)
        x = x.permute(0, 5, 4, 1, 2, 3).contiguous()
        # tansform dims shape from (B,C,T,H,W,D) to (B,C,H,W,D)
        x = x.reshape(B,C,H*self.patch_size,W*self.patch_size,D*self.patch_size)
        # print(x.shape)
        return x

    def forward(self, x):
        x = self.group_time_mixing(x)
        return x


if __name__ == "__main__":
    # shift1 = shift_Windows(ty="shift_token", S=2, C=32)
    # shift1(x)

    B = 64
    C = 3
    W=H=D =32
    x = torch.randn(B * C * H * W * D)
    x = torch.reshape(x, (B, C, H, W, D))
    print(x.shape)
    # init shift method
    # shift1 = shift_Windows(ty="shift_token", S=2, C=C,patch_size=8)
    # #  process x
    # x = shift1(x)


class Mlp_UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes,args=None):
        super(Mlp_UNet3D, self).__init__()
        # init shift method
        # ty = args.model_name
        ty = "shift_token"
        self.shift0 = shift_Windows(ty=ty, S=2, C=32,patch_size=8)
        self.shift1 = shift_Windows(ty=ty, S=2, C=32,patch_size=8)
        self.shift2 = shift_Windows(ty=ty, S=2, C=64,patch_size=4)

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

        # output layer: ?x32x32x32x32 ==> ?x32x32x32x4# full assembly of the sub-parts to form the complete net
        # import math
        #
        # import torch.nn.functional as F
        # import torch
        # import torch.nn as nn
        # from torch.nn.init import trunc_normal_
        #
        # from .unet_parts import *
        # # from unet_parts import *
        # from patchify import patchify
        # from timm.models.layers import DropPath
        #
        #
        #
        # class shift_Windows(nn.Module):
        #     def __init__(self, ty=None, S=32, C=32,patch_size=2):
        #         super(shift_Windows, self).__init__()
        #         self.ty = ty
        #         self.S = S
        #         self.C = C
        #         self.patch_size = patch_size
        #         if ty == "shift_token":
        #             self.linear = nn.Linear(S * C, C)
        #         else:
        #             self.linear = nn.Linear(S * C, S * C)
        #
        #     def shift(self,x,strid):
        #         # input x (B,H,W,D,T,C)
        #         # shift roll T dim
        #         return torch.roll(x,shifts=(strid),dims=(4))
        #     '''
        #         input x shape (B,C,T,H,W,D)
        #         need x shape  (B,H,W,T,C)
        #     '''
        #     def group_time_mixing(self, x):
        #         B, C,H, W, D = x.shape
        #         self.patch_size=H
        #         self.linear=nn.Linear(2*C,C)
        #         # get patches
        #         x = x.reshape(B, C, -1, torch.div(H,self.patch_size,rounding_mode="floor"), W, D)
        #         x = x.permute(0, 3, 4, 5, 2, 1).contiguous()
        #         B,H,W,D,T,C = x.shape
        #         if self.ty == "short_range":
        #             x = self.linear(x.reshape(B,H,W,D, -1, self.S * C))
        #             x = x.reshape(B,H,W,D,T,C)
        #         elif self.ty == "long_range":
        #             x = x.reshape(B,H,W,D,self.S,-1,C).transpose(4,5)
        #             x = self.linear(x.reshape(B,H,W,D,-1,self.S*C))
        #             x = x.reshape(B,H,W,D,-1,self.S,C).transpose(4,5)
        #             x = x.reshape(B,H,W,D,T,C)
        #         elif self.ty == "shift_window":
        #             x = self.shift(x,torch.div(self.S,2,rounding_mode="floor"))
        #             x=self.linear(x.reshape(B,H,W,D,-1,self.S*C))
        #             x= self.shift(x.reshape(B,H,W,D,T,C),torch.div(-self.S,2,rounding_mode="floor"))
        #         elif self.ty == "shift_token":
        #             x = [self.shift(x,i) for i in range(self.S)]
        #             x = self.linear(torch.cat(x, dim=5))
        #
        #         # tansform dims shape from (B,H,W,D,T,C) to (B,C,T,H,W,D)
        #         x = x.permute(0, 5, 4, 1, 2, 3).contiguous()
        #         # tansform dims shape from (B,C,T,H,W,D) to (B,C,H,W,D)
        #         x = x.reshape(B,C,H*self.patch_size,W,D)
        #         # print(x.shape)
        #         return x
        #
        #     def forward(self, x,W,D):
        #         """
        #         input B N H C
        #         output B N H C
        #         """
        #         B, _, H, C = x.shape
        #         x = x.transpose(1, 3).view(B, C, H, W, D).contiguous()
        #         x = self.group_time_mixing(x)
        #
        #         x = x.reshape(B, C, H, W * D).contiguous()
        #         x = x.transpose(1, 3)
        #         return x
        #
        #
        # class OverlapPatchEmbed(nn.Module):
        #     """ Image to Patch Embedding
        #     """
        #
        #     def __init__(self, patch_size=2, stride=2, in_chans=3, embed_dim=128):
        #         super().__init__()
        #         self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=1)
        #
        #         self.norm = nn.LayerNorm(embed_dim)
        #
        #         self.apply(self._init_weights)
        #
        #     def _init_weights(self, m):
        #         if isinstance(m, nn.Linear):
        #             trunc_normal_(m.weight, std=.02)
        #             if isinstance(m, nn.Linear) and m.bias is not None:
        #                 nn.init.constant_(m.bias, 0)
        #         elif isinstance(m, nn.LayerNorm):
        #             nn.init.constant_(m.bias, 0)
        #             nn.init.constant_(m.weight, 1.0)
        #         elif isinstance(m, nn.Conv2d):
        #             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #             fan_out //= m.groups
        #             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        #             if m.bias is not None:
        #                 m.bias.data.zero_()
        #
        #     def forward(self, x):
        #         x = self.proj(x)  # x :(B,C,H,W,D)
        #         _, _, H, W, D = x.shape
        #         x = x.flatten(3).transpose(1, 3)  # x : (B,W*D,H,C)
        #         x = self.norm(x)
        #
        #         return x, H, W, D
        #
        #
        # class DWConv(nn.Module):
        #     def __init__(self, dim=128):
        #         super(DWConv, self).__init__()
        #         self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        #
        #     def forward(self, x, W, D):
        #         B, N, H, C = x.shape
        #         x = x.transpose(1, 3).view(B, C, H, W, D)
        #         x = self.dwconv(x)
        #         x = x.flatten(3).transpose(1, 3)
        #
        #         return x
        #
        #
        # class shiftmlp(nn.Module):
        #     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        #         super().__init__()
        #         out_features = out_features or in_features
        #         hidden_features = hidden_features or in_features
        #         self.dim = in_features
        #         self.fc1 = nn.Linear(in_features, hidden_features)
        #         self.dwconv = DWConv(hidden_features)
        #         self.act = act_layer()
        #         self.fc2 = nn.Linear(hidden_features, out_features)
        #         self.drop = nn.Dropout(drop)
        #
        #         self.shift_size = shift_size
        #         self.pad = shift_size // 2
        #
        #         self.apply(self._init_weights)
        #
        #     def _init_weights(self, m):
        #         if isinstance(m, nn.Linear):
        #             trunc_normal_(m.weight, std=.02)
        #             if isinstance(m, nn.Linear) and m.bias is not None:
        #                 nn.init.constant_(m.bias, 0)
        #         elif isinstance(m, nn.LayerNorm):
        #             nn.init.constant_(m.bias, 0)
        #             nn.init.constant_(m.weight, 1.0)
        #         elif isinstance(m, nn.Conv2d):
        #             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #             fan_out //= m.groups
        #             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        #             if m.bias is not None:
        #                 m.bias.data.zero_()
        #
        #     def forward(self, x, W, D):
        #         # pdb.set_trace()
        #         B, N, H, C = x.shape  # x:(B,W*D,H,C)
        #
        #         xn = x.transpose(1, 3).view(B, C, H, W, D).contiguous()  # xn: (8,160,32,32)
        #         ### shift domain
        #         xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)  # xn:(8,160,36,36)
        #         xs = torch.chunk(xn, self.shift_size, 1)  # seplit xn along 1 axis  # x_shift[0]:(8,32,36,36)
        #         x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        #         ###tokenize domain
        #         x_cat = torch.cat(x_shift, 1)
        #         x_cat = torch.narrow(x_cat, 3, self.pad, W)
        #         x_s = torch.narrow(x_cat, 4, self.pad, D)
        #         ###tokens doamin
        #         x_s = x_s.reshape(B, C, H, W * D).contiguous()
        #         x_shift_r = x_s.transpose(1, 3)
        #
        #         x = self.fc1(x_shift_r)  # Linear
        #         ### DW+GELU doamin
        #         x = self.dwconv(x, W, D)
        #         x = self.act(x)  # GELU
        #         x = self.drop(x)
        #
        #         xn = x.transpose(1, 3).view(B, C, H, W, D).contiguous()
        #         ###shift domain
        #         xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        #         xs = torch.chunk(xn, self.shift_size, 1)
        #         x_shift = [torch.roll(x_c, shift, 4) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        #         ###tokenized domain
        #         x_cat = torch.cat(x_shift, 1)
        #         x_cat = torch.narrow(x_cat, 3, self.pad, W)
        #         x_s = torch.narrow(x_cat, 4, self.pad, D)
        #         ### tokens domian
        #         x_s = x_s.reshape(B, C, H, W * D).contiguous()
        #         x_shift_c = x_s.transpose(1, 3)
        #         ###MLP domain
        #         x = self.fc2(x_shift_c)
        #         x = self.drop(x)
        #         return x
        #
        #
        # class shiftedBlock(nn.Module):
        #     def __init__(self, dim, mlp_ratio=1., drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        #         super().__init__()
        #         self.drop_path = nn.Identity()
        #         self.norm2 = norm_layer(dim)
        #         mlp_hidden_dim = int(dim * mlp_ratio)
        #         self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        #         self.apply(self._init_weights)
        #
        #         ## ty
        #         self.ty = "shift_token"
        #         self.S=2
        #         self.C=dim
        #         if self.ty == "shift_token":
        #             self.linear = nn.Linear(self.S * self.C, self.C)
        #         else:
        #             self.linear = nn.Linear(self.S * self.C, self.S * self.C)
        #         self.shift_linear = nn.Linear(dim,dim)
        #
        #     def _init_weights(self, m):
        #         if isinstance(m, nn.Linear):
        #             trunc_normal_(m.weight, std=.02)
        #             if isinstance(m, nn.Linear) and m.bias is not None:
        #                 nn.init.constant_(m.bias, 0)
        #         elif isinstance(m, nn.LayerNorm):
        #             nn.init.constant_(m.bias, 0)
        #             nn.init.constant_(m.weight, 1.0)
        #         elif isinstance(m, nn.Conv2d):
        #             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #             fan_out //= m.groups
        #             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        #             if m.bias is not None:
        #                 m.bias.data.zero_()
        #     def shift(self,x,strid):
        #         # input x (B,H,W,D,T,C)
        #         # shift roll T dim
        #         return torch.roll(x,shifts=(strid),dims=(4))
        #     '''
        #         input x shape (B,C,T,H,W,D)
        #         need x shape  (B,H,W,T,C)
        #     '''
        #     def group_time_mixing(self, x):
        #         B, C,H, W, D = x.shape
        #         self.patch_size=H
        #         # get patches
        #         x = x.reshape(B, C, -1, torch.div(H,self.patch_size,rounding_mode="floor"), W, D)
        #         x = x.permute(0, 3, 4, 5, 2, 1).contiguous()
        #         B,H,W,D,T,C = x.shape
        #         if self.ty == "short_range":
        #             x = self.linear(x.reshape(B,H,W,D, -1, self.S * C))
        #             x = x.reshape(B,H,W,D,T,C)
        #         elif self.ty == "long_range":
        #             x = x.reshape(B,H,W,D,self.S,-1,C).transpose(4,5)
        #             x = self.linear(x.reshape(B,H,W,D,-1,self.S*C))
        #             x = x.reshape(B,H,W,D,-1,self.S,C).transpose(4,5)
        #             x = x.reshape(B,H,W,D,T,C)
        #         elif self.ty == "shift_window":
        #             x = self.shift(x,torch.div(self.S,2,rounding_mode="floor"))
        #             x=self.linear(x.reshape(B,H,W,D,-1,self.S*C))
        #             x= self.shift(x.reshape(B,H,W,D,T,C),torch.div(-self.S,2,rounding_mode="floor"))
        #         elif self.ty == "shift_token":
        #             x = [self.shift(x,i) for i in range(self.S)]
        #             x = self.linear(torch.cat(x, dim=5))
        #
        #         # tansform dims shape from (B,H,W,D,T,C) to (B,C,T,H,W,D)
        #         x = x.permute(0, 5, 4, 1, 2, 3).contiguous()
        #         # tansform dims shape from (B,C,T,H,W,D) to (B,C,H,W,D)
        #         x = x.reshape(B,C,H*self.patch_size,W,D)
        #         # print(x.shape)
        #         return x
        #
        #     def H_shift(self, x,W,D):
        #         """
        #         input B N H C
        #         output B N H C
        #         """
        #         B, _, H, C = x.shape
        #         x = x.transpose(1, 3).view(B, C, H, W, D).contiguous()
        #         x = self.group_time_mixing(x)
        #
        #         x = x.reshape(B, C, H, W * D).contiguous()
        #         x = x.transpose(1, 3)
        #         return x
        #
        #     def forward(self, x, W, D):
        #         # W D axis shift
        #         W_D_Shift = self.mlp(self.norm2(x), W, D)
        #
        #         # _,_,H,_=x.shape
        #         #H_shift =self.shift_T(ty=self.ty, S=2, C=self.channel, patch_size=H)
        #         H_W_D_Shift =self.H_shift(x, W, D)
        #         x = x + self.drop_path(self.shift_linear(H_W_D_Shift))
        #         x = x + self.drop_path(H_W_D_Shift)
        #         return x
        #
        #
        # class Mlp_UNet3D(nn.Module):
        #     def __init__(self, n_channels, n_classes, args=None):
        #         super(Mlp_UNet3D, self).__init__()
        #
        #         # input converlution: ?x32x32x32x1 ==> ?x32x32x32x32
        #         self.dow_conv1 = inconv(n_channels, 16)
        #         self.dow_conv2 = inconv(16, 32)
        #         self.dow_conv3 = inconv(32, 64)
        #
        #         self.Bn_dow_conv1 = nn.BatchNorm3d(16)
        #         self.Bn_dow_conv2 = nn.BatchNorm3d(32)
        #         self.Bn_dow_conv3 = nn.BatchNorm3d(64)
        #
        #         self.patch_embed1 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=64, embed_dim=128)
        #         self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=128, embed_dim=256)
        #
        #         self.block1 = shiftedBlock(dim=128, mlp_ratio=1)
        #         self.block2 = shiftedBlock(dim=256, mlp_ratio=1)
        #
        #         self.dblock1 = shiftedBlock(dim=128, mlp_ratio=1)
        #         self.dblock2 = shiftedBlock(dim=64, mlp_ratio=1)
        #
        #         self.norm1 = nn.LayerNorm(128)
        #         self.norm2 = nn.LayerNorm(256)
        #
        #         self.dnorm1 = nn.LayerNorm(128)
        #         self.dnorm2 = nn.LayerNorm(64)
        #
        #         self.decoder1 = nn.Conv3d(256, 128, 3, stride=1, padding=1)
        #         self.decoder2 = nn.Conv3d(128, 64, 3, stride=1, padding=1)
        #         self.decoder3 = nn.Conv3d(64, 32, 3, stride=1, padding=1)
        #         self.decoder4 = nn.Conv3d(32, 16, 3, stride=1, padding=1)
        #         self.decoder5 = nn.Conv3d(16, 4, 3, stride=1, padding=1)
        #
        #         self.dbn1 = nn.BatchNorm3d(128)
        #         self.dbn2 = nn.BatchNorm3d(64)
        #         self.dbn3 = nn.BatchNorm3d(32)
        #         self.dbn4 = nn.BatchNorm3d(16)
        #         self.dbn5 = nn.BatchNorm3d(4)
        #
        #         self.final = nn.Conv3d(4, n_classes, kernel_size=1)
        #
        #     def forward(self, x):
        #         B = x.shape[0]
        #         # input converlution stage
        #         out = F.relu6(F.max_pool3d(self.Bn_dow_conv1(self.dow_conv1(x)), 2, 2))
        #         t1 = out
        #         out = F.relu6(F.max_pool3d(self.Bn_dow_conv2(self.dow_conv2(out)), 2, 2))
        #         t2 = out
        #         out = F.relu6(F.max_pool3d(self.Bn_dow_conv3(self.dow_conv3(out)), 2, 2))
        #         t3 = out
        #
        #         ### Tokenized MLP Stage
        #         ### Token Mlp domain 1
        #         out, H, W, D = self.patch_embed1(out)  # out:(64,9,3,128) B,H,D*W,C
        #         out = self.block1(out,W, D)
        #         out = self.norm1(out)
        #         out = out.reshape(B, W, D, H, -1).permute(0, 4, 3, 1, 2).contiguous()
        #         t4 = out
        #         # Token Mlp domain 2
        #         out, H, W, D = self.patch_embed2(out)  # out:(64,9,3,128) B,H,D*W,C
        #         out = self.block2(out, W, D)
        #         out = self.norm2(out)
        #         out = out.reshape(B, W, D, H, -1).permute(0, 4, 3, 1, 2).contiguous()
        #
        #         ### up stage
        #         # up1
        #         out = F.relu6(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        #         out = torch.add(out, t4)
        #         _, _, H, W, D = out.shape
        #         out = out.flatten(3).transpose(1, 3)
        #         out = self.dblock1(out, W, D)
        #         out = self.dnorm1(out)
        #         out = out.reshape(B, W, D, H, -1).permute(0, 4, 3, 1, 2).contiguous()
        #         # up 2
        #         out = F.relu6(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        #         out = torch.add(out, t3)
        #         _, _, H, W, D = out.shape
        #         out = out.flatten(3).transpose(1, 3)
        #         out = self.dblock2(out, W, D)
        #         out = self.dnorm2(out)
        #         out = out.reshape(B, W, D, H, -1).permute(0, 4, 3, 1, 2).contiguous()
        #         #######################################################
        #         # up 3
        #         out = F.relu6(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        #         out = torch.add(out, t2)
        #         out = F.relu6(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        #         out = torch.add(out, t1)
        #         out = F.relu6(F.interpolate(self.dbn5(self.decoder5(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        #         out = self.final(out)
        #
        #         return out
        self.outc = outconv(32, n_classes)

    def forward(self, x):
        # input converlution layer
        x0 = self.inc(x=x)
        # down layers
        x0 = self.shift0(x0)
        x1 = self.down0(x=x0)

        x1 = self.shift1(x1)
        x2 = self.down1(x=x1)

        x2 = self.shift2(x2)
        x3 = self.down2(x=x2)
        # up layers
        x4 = self.up0(x=x3, skip_connct=x2)
        x5 = self.up1(x=x4, skip_connct=x1)
        # output layer
        x6 = self.outc(x5)

        return x6
