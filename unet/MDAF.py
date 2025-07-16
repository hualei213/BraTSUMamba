# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
#
# # Patch Embedding to reduce token count
# class PatchEmbedding3D(nn.Module):
#     def __init__(self, in_channels, patch_size, out_channels):
#         super(PatchEmbedding3D, self).__init__()
#         self.patch_size = patch_size
#         # Calculate stride dynamically based on input dimensions and patch size
#         self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size)
#
#     def forward(self, x):
#         # Perform patch embedding via a 3D convolution with stride == patch size
#         return self.conv(x)
#
#
# def to_3d(x):
#     return rearrange(x, 'b c h w d -> b (h w d) c')
#
# def to_4d(x, h, w, d):
#     return rearrange(x, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)
#
# class BiasFree_LayerNorm(nn.Module):
#     def __init__(self, normalized_shape):
#         super(BiasFree_LayerNorm, self).__init__()
#         if isinstance(normalized_shape, int):
#             normalized_shape = (normalized_shape,)
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.normalized_shape = normalized_shape
#
#     def forward(self, x):
#         sigma = x.var(-1, keepdim=True, unbiased=False)
#         return x / torch.sqrt(sigma + 1e-5) * self.weight
#
#
# class WithBias_LayerNorm(nn.Module):
#     def __init__(self, normalized_shape):
#         super(WithBias_LayerNorm, self).__init__()
#         if isinstance(normalized_shape, int):
#             normalized_shape = (normalized_shape,)
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.normalized_shape = normalized_shape
#
#     def forward(self, x):
#         mu = x.mean(-1, keepdim=True)
#         sigma = x.var(-1, keepdim=True, unbiased=False)
#         return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
#
#
# class LayerNorm(nn.Module):
#     def __init__(self, dim, LayerNorm_type):
#         super(LayerNorm, self).__init__()
#         # 确保根据输入维度动态初始化
#         if LayerNorm_type == 'BiasFree':
#             self.body = BiasFree_LayerNorm(dim)
#         else:
#             self.body = WithBias_LayerNorm(dim)
#
#     def forward(self, x):
#         # 确保根据输入维度进行调整
#         h, w, d = x.shape[-3:]  # Extract height, width, and depth dimensions
#         return to_4d(self.body(to_3d(x)), h, w, d)
#
#
# class MDAF(nn.Module):
#     def __init__(self, dim, num_heads, LayerNorm_type, patch_size=(4, 4, 4)):
#         super(MDAF, self).__init__()
#         self.num_heads = num_heads
#
#         # Patch embedding layers to reduce the spatial resolution
#         self.patch_embed1 = PatchEmbedding3D(dim, patch_size, dim)
#         self.patch_embed2 = PatchEmbedding3D(dim, patch_size, dim)
#
#         # LayerNorm initialization
#         self.norm1 = LayerNorm(dim, LayerNorm_type)
#         self.norm2 = LayerNorm(dim, LayerNorm_type)
#
#         # Convolution layers as per the diagram
#         self.conv3x3 = nn.Conv3d(dim, dim, kernel_size=3, padding=1)
#         self.conv5x5 = nn.Conv3d(dim, dim, kernel_size=5, padding=2)
#         self.conv7x7 = nn.Conv3d(dim, dim, kernel_size=7, padding=3)
#         self.conv1x1 = nn.Conv3d(dim, dim, kernel_size=1)
#
#         self.deconv = nn.ConvTranspose3d(dim, dim, kernel_size=patch_size, stride=patch_size)
#
#         # Final projection layer
#         self.project_out = nn.Conv3d(dim, dim, kernel_size=1)
#
#     def forward(self, x1, x2):
#         # Normalize the inputs
#         x1_orgin = self.norm1(x1)
#         x2_orgin = self.norm2(x2)
#
#         # Step 1: Apply Patch Embedding (downsampling)
#         x1 = self.patch_embed1(x1_orgin)  # Downsample input 1
#         x2 = self.patch_embed2(x2_orgin)  # Downsample input 2
#
#         b, c, h, w, d = x1.shape
#
#         # Step 2: Apply convolutions with different kernel sizes
#         conv3x3_out1 = self.conv3x3(x1)
#         conv5x5_out1 = self.conv5x5(x1)
#         conv7x7_out1 = self.conv7x7(x1)
#
#         conv3x3_out2 = self.conv3x3(x2)
#         conv5x5_out2 = self.conv5x5(x2)
#         conv7x7_out2 = self.conv7x7(x2)
#
#         # Sum the outputs
#         out1 = conv3x3_out1 + conv5x5_out1 + conv7x7_out1
#         out2 = conv3x3_out2 + conv5x5_out2 + conv7x7_out2
#
#         # Apply the 1x1 convolution
#         out1 = self.conv1x1(out1)
#         out2 = self.conv1x1(out2)
#
#         # Step 3: Perform attention mechanism
#         k1 = rearrange(out1, 'b (head c) h w d -> b head h w (d c)', head=self.num_heads)
#         v1 = rearrange(out1, 'b (head c) h w d -> b head h w (d c)', head=self.num_heads)
#         q1 = rearrange(out2, 'b (head c) h w d -> b head w h (d c)', head=self.num_heads)
#
#         # Normalize queries and keys
#         q1 = F.normalize(q1, dim=-1)
#         k1 = F.normalize(k1, dim=-1)
#
#         # Attention computations
#         attn1 = (q1 @ k1.transpose(-2, -1))
#         attn1 = attn1.softmax(dim=-1)
#         out3 = (attn1 @ v1) + q1
#
#         # Reshape the output back to original dimensions
#         out3 = rearrange(out3, 'b head h w (d c) -> b (head c) h w d', head=self.num_heads, h=h, w=w, d=d)
#
#         out3 = self.deconv(out3)
#
#         # Final projection and return the result
#         out = self.project_out(out3) + x1_orgin
#
#         return out
#
#
# if __name__ == "__main__":
#     img = torch.ones([1, 64, 128, 128,128]).cuda()  # Change size based on your input shape
#
#     model = MDAF(64, num_heads=8, LayerNorm_type='WithBias')
#     model = model.cuda()
#     parameters = filter(lambda p: p.requires_grad, model.parameters())
#     parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
#     print('Trainable Parameters: %.3fM' % parameters)
#
#     out_img = model(img, img)
#
#     print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]
#     from thop import profile
#
#     flops, params = profile(model, (img, img))
#     print('flops:', flops, 'params:', params)
#     print('flops:%.2f G,params: %.2f M' % (flops / 1e9, params / 1e6))


















import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w d -> b (h w d) c')

def to_4d(x, h, w, d):
    return rearrange(x, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w, d = x.shape[-3:]  # Extract height, width, and depth dimensions
        return to_4d(self.body(to_3d(x)), h, w, d)


class MDAF(nn.Module):
    def __init__(self, dim, num_heads, LayerNorm_type):
        super(MDAF, self).__init__()
        self.num_heads = num_heads

        # LayerNorm initialization
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)

        # Convolution layers as per the diagram
        self.conv3x3 = nn.Conv3d(dim, dim, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv3d(dim, dim, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv3d(dim, dim, kernel_size=7, padding=3)
        self.conv1x1 = nn.Conv3d(dim, dim, kernel_size=1)

        # Final projection layer
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1)

    def forward(self, x1, x2):
        b, c, h, w, d = x1.shape
        # Normalize the inputs
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)

        # Apply convolutions with different kernel sizes
        conv3x3_out1 = self.conv3x3(x1)
        conv5x5_out1 = self.conv5x5(x1)
        conv7x7_out1 = self.conv7x7(x1)

        conv3x3_out2 = self.conv3x3(x2)
        conv5x5_out2 = self.conv5x5(x2)
        conv7x7_out2 = self.conv7x7(x2)

        # Sum the outputs
        out1 = conv3x3_out1 + conv5x5_out1 + conv7x7_out1
        out2 = conv3x3_out2 + conv5x5_out2 + conv7x7_out2
        # Apply the 1x1 convolution
        out1 = self.conv1x1(out1)
        out2 = self.conv1x1(out2)


        # Perform attention mechanism (same as before)
        k1 = rearrange(out1, 'b (head c) h w d -> b head h w (d c)', head=self.num_heads)
        v1 = rearrange(out1, 'b (head c) h w d -> b head h w (d c)', head=self.num_heads)

        # k2 = rearrange(out2, 'b (head c) h w d -> b head w h (d c)', head=self.num_heads)
        # v2 = rearrange(out2, 'b (head c) h w d -> b head w h (d c)', head=self.num_heads)
        #
        # q2 = rearrange(out1, 'b (head c) h w d -> b head h w (d c)', head=self.num_heads)
        q1 = rearrange(out2, 'b (head c) h w d -> b head w h (d c)', head=self.num_heads)

        # Normalize queries and keys
        q1 = F.normalize(q1, dim=-1)
        # q2 = F.normalize(q2, dim=-1)
        k1 = F.normalize(k1, dim=-1)
        # k2 = F.normalize(k2, dim=-1)

        # Attention computations
        attn1 = (q1 @ k1.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        out3 = (attn1 @ v1) + q1

        # attn2 = (q2 @ k2.transpose(-2, -1))
        # attn2 = attn2.softmax(dim=-1)
        # out4 = (attn2 @ v2) + q2

        # Reshape the output back to original dimensions
        out3 = rearrange(out3, 'b head h w (d c) -> b (head c) h w d', head=self.num_heads, h=h, w=w, d=d)
        # out4 = rearrange(out4, 'b head w h (d c) -> b (head c) h w d', head=self.num_heads, h=h, w=w, d=d)

        # Project and return the result
        out = self.project_out(out3) + x1

        return out

if __name__ == "__main__":
    img = torch.ones([1, 128, 128, 128, 128]).cuda()

    model = MDAF(128, num_heads=8, LayerNorm_type='WithBias')
    model = model.cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out_img = model(img,img)

    print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]
    from thop import profile

    flops, params = profile(model, (img,img))
    print('flops:', flops, 'params:', params)
    print('flops:%.2f G,params: %.2f M' % (flops / 1e9, params / 1e6))