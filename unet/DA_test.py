import torch
import torch.nn as nn
from torch.nn import Module, Parameter, Softmax


class DepthWiseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(DepthWiseConv3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1, 1, 0, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class PAM_Module(Module):
    """ Position attention module for 3D inputs """

    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = DepthWiseConv3d(in_dim, in_dim, kernel_size=1)
        self.key_conv = DepthWiseConv3d(in_dim, in_dim, kernel_size=1)
        self.value_conv = DepthWiseConv3d(in_dim, in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, D, H, W = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, D * H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, D * H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, D * H * W)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, D, H, W)
        out = self.gamma * out + x
        return out


class CAM_Module(Module):
    """ Channel attention module for 3D inputs """

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, D, H, W = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, D, H, W)

        out = self.gamma * out + x
        return out


class DA_Block(nn.Module):
    def __init__(self, in_channels):
        super(DA_Block, self).__init__()
        inter_channels = in_channels // 16

        self.conv5a = nn.Sequential(DepthWiseConv3d(in_channels, inter_channels, 3, padding=1), nn.ReLU())
        self.conv5c = nn.Sequential(DepthWiseConv3d(in_channels, inter_channels, 3, padding=1), nn.ReLU())
        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(DepthWiseConv3d(inter_channels, inter_channels, 3, padding=1), nn.ReLU())
        self.conv52 = nn.Sequential(DepthWiseConv3d(inter_channels, inter_channels, 3, padding=1), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Dropout3d(0.05, False), DepthWiseConv3d(inter_channels, in_channels, 1), nn.ReLU())
        self.conv7 = nn.Sequential(nn.Dropout3d(0.05, False), DepthWiseConv3d(inter_channels, in_channels, 1), nn.ReLU())
        self.conv8 = nn.Sequential(nn.Dropout3d(0.05, False), DepthWiseConv3d(in_channels, in_channels, 1), nn.ReLU())

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output1 = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output2 = self.conv7(sc_conv)

        feat_sum = sa_output1 + sc_output2
        sasc_output = self.conv8(feat_sum)

        return sasc_output


class ModelWithDABlocks(nn.Module):
    def __init__(self):
        super(ModelWithDABlocks, self).__init__()
        self.block1 = DA_Block(128)  # For input (B, 128, 8, 8, 8)
        self.block2 = DA_Block(64)   # For input (B, 64, 16, 16, 16)
        self.block3 = DA_Block(32)   # For input (B, 32, 32, 32, 32)
        self.block4 = DA_Block(16)   # For input (B, 16, 64, 64, 64)

    def forward(self, inputs):
        x = self.block1(inputs)  # Process (B, 128, 8, 8, 8)
        # x2 = self.block2(inputs[1])  # Process (B, 64, 16, 16, 16)
        # x3 = self.block3(inputs[2])  # Process (B, 32, 32, 32, 32)
        # x4 = self.block4(inputs[3])  # Process (B, 16, 64, 64, 64)
        return x      # Return all outputs


if __name__ == "__main__":
    model = ModelWithDABlocks()
    input1 = torch.randn(2, 128, 8, 8, 8)
    input2 = torch.randn(2, 64, 16, 16, 16)
    input3 = torch.randn(2, 32, 32, 32, 32)
    input4 = torch.randn(2, 16, 64, 64, 64)

    output = model(input1)
    # for i, output in enumerate(outputs):
    print(f"Output shape: {output.shape}")