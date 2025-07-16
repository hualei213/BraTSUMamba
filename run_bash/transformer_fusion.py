from functools import reduce
from operator import mul
from timm.models.layers import DropPath, trunc_normal_
import torch.nn as nn
import torch.nn.functional as F


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)

def window_partition(x, window_size: int):
        """
        将feature map按照window_size划分成一个个没有重叠的window
        Args:
            x: (B, H, W, D, C)
            window_size (int): window size(M)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, D, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, D // window_size, window_size, C)  # (2,2,3,2,3,8)
        # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
        # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)  # (8,3,3,8)
        return windows


def window_reverse(windows, window_size, B, H, W, D):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, H // window_size[0], W // window_size[1],  D // window_size[2],window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


class PatchPart_LinearEmbedding(nn.Moudle):


    def __int__(self,in_chanel=4,embed_dim=96,patch_size=(4,4,4)):
        super().__int__()
        self.patch_size = patch_size
        self.in_chanel = in_chanel
        self.project = nn.Conv3d(in_chanel,embed_dim,kernel_size=patch_size,stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)


    def forward(self,x):
        B,C,H,W,D = x.size()
        x = self.project(x)
        x = x.flatten(2).transpose(1,2)
        x = self.norm(x)
        x = x.transpose(1, 2).view(-1, self.embed_dim, H, W, D)

        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # x2 = None

        # if is_decoder:
        #     q = q * self.scale
        #     attn2 = q @ prev_k.transpose(-2, -1)
        #     attn2 = attn2 + relative_position_bias.unsqueeze(0)
        #
        #     if mask is not None:
        #         nW = mask.shape[0]
        #         attn2 = attn2.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        #         attn2 = attn2.view(-1, self.num_heads, N, N)
        #         attn2 = self.softmax(attn2)
        #     else:
        #         attn2 = self.softmax(attn2)
        #
        #     attn2 = self.attn_drop(attn2)
        #
        #     x2 = (attn2 @ prev_v).transpose(1, 2).reshape(B_, N, C)
        #     x2 = self.proj(x2)
        #     x2 = self.proj_drop(x2)

        # return x, x2, v, k, q
        return x, v, k, q


class SwinTransformer3D(nn.Module):
    def __int__(self,dim,num_heads,window_size=(7,7,7), shift_size = (0, 0, 0),
        mlp_ratio = 4., qkv_bias = True, qk_scale = None, drop = 0., attn_drop = 0., drop_path = 0., act_layer = nn.GELU, norm_layer = nn.LayerNorm, use_checkpoint = False, patch_size = 16, stride = 1):
        super().__int__()

        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)




        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()





    def W_MSA_SW_MSA(self,x,mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        pad_d1 = (window_size[2] - D % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape

        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None


        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows, v, k, q = self.attn(x_windows,mask=attn_mask)  # B*nW, Wd*Wh*Ww, C

        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Hp, Wp,Dp)  # B D' H' W' C

        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        x2 = None
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()

        # if cross_attn_windows is not None:
        #     # merge windows
        #     cross_attn_windows = cross_attn_windows.view(-1, *(window_size + (C,)))
        #     cross_shifted_x = window_reverse(cross_attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        #     # reverse cyclic shift
        #     if any(i > 0 for i in shift_size):
        #         x2 = torch.roll(cross_shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        #     else:
        #         x2 = cross_shifted_x
        #
        #     if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
        #         x2 = x2[:, :D, :H, :W, :].contiguous()

        return x, x2, v, k, q


    def LN_MLP(self,x):
        return self.drop_path(self.mlp(self.norm2(x)))






    def forward(self, x, mask_matrix, prev_v, prev_k, prev_q, is_decoder=False):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        alpha = 0.5
        shortcut = x
        x2, v, k, q = None, None, None, None

        # if self.use_checkpoint:
        #     x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        # else:
        x, x2, v, k, q = self.W_MSA_SW_MSA(x, mask_matrix, prev_v, prev_k, prev_q, is_decoder)

        #FFN
        x = shortcut + self.drop_path(x)
        #Norm and MLP
        # if self.use_checkpoint:
        #     x = x + checkpoint.checkpoint(self.forward_part2, x)
        # else:
        # x = x + self.forward_part2(x)
        x = x + self.LN_MLP(x)



        if x2 is not None:
            x2 = shortcut + self.drop_path(x2)
            # if self.use_checkpoint:
            #     x2 = x2 + checkpoint.checkpoint(self.forward_part2, x2)
            # else:
            x2 = x2 + self.forward_part2(x2)

            FPE = PositionalEncoding3D(x.shape[4])

            x = torch.add((1 - alpha) * x, alpha * x2) + self.forward_part3(FPE(x))

        return x, v, k, q





class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 depth,
                 depths,
                 num_heads,
                 window_size=(1, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x, block_num):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')

        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

        v1, k1, q1, v2, k2, q2 = None, None, None, None, None, None

        for idx, blk in enumerate(self.blocks):
            if idx % 2 == 0:
                x, v1, k1, q1 = blk(x, attn_mask, None, None, None)
            else:
                x, v2, k2, q2 = blk(x, attn_mask, None, None, None)

        x = x.reshape(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')

        return x, v1, k1, q1, v2, k2, q2











