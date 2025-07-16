import torch.nn.functional as F
import torch
import torch.nn as nn

from timm.models.layers import DropPath


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
        # self.drop = DropPath(0.1)
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
        X_out = X_out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        # X_out = self.drop(X_out)
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


class MLP_Vnet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2):
        super(MLP_Vnet, self).__init__()
        # down sample
        self.dow_conv1 = nn.Conv3d(in_channels=n_channels, out_channels=16, kernel_size=1, padding=0, stride=1,
                                   bias=True)
        self.dow_conv2 = down_residual_conv_block(in_channels=16, out_channels=32)
        # MLP stage
        self.dow_mlp_block1 = down_MLP_Block(in_channel=32, out_channel=64, tokenize_number=384)
        self.dow_mlp_block2 = down_MLP_Block(in_channel=64, out_channel=128, tokenize_number=196)
        self.dow_mlp_block3 = down_MLP_Block(in_channel=128, out_channel=256, tokenize_number=98)

        self.up_mlp_block1 = up_MLP_Block(in_channel=256, out_channel=128, tokenize_number=196)
        self.up_mlp_block2 = up_MLP_Block(in_channel=128, out_channel=64, tokenize_number=384)
        self.up_conv3 = up_residual_conv_block(in_channels=64, out_channels=32)
        self.up_conv4 = up_residual_conv_block(in_channels=32, out_channels=16)
        self.up_conv5 = nn.ConvTranspose3d(in_channels=16, out_channels=n_classes, kernel_size=1, padding=0, stride=1,
                                           bias=True)

    def forward(self, x):
        # down stage
        out = self.dow_conv1(x)
        x1 = out
        out = self.dow_conv2(out)
        x2 = out
        # mlp mixer stage
        x3_out, T3_in = self.dow_mlp_block1(out)
        x4_out, T4_in = self.dow_mlp_block2(x3_out)
        x5_out, T5_in = self.dow_mlp_block3(x4_out)

        ## deconder stage
        out = self.up_mlp_block1(x5_out, T4_in)
        out = self.up_mlp_block2(out, T3_in)

        out = self.up_conv3(out)
        out = torch.add(out, x2)

        out = self.up_conv4(out)
        out = torch.add(out, x1)
        # final
        out = F.softmax(self.up_conv5(out), dim=1)

        return out


if __name__ == "__main__":
    device = torch.device('cpu')
    input_data = torch.randn((1, 1, 128, 128, 128))
    model = MLP_Vnet(1, 2)
    model.to(device)
    input_data = input_data.to(device)
    output_data = model(input_data)
    print(output_data.shape)
