import torch.nn as nn
import torch.nn.functional as F
import torch


class BinaryDice(nn.Module):
    def __init__(self):
        super(BinaryDice, self).__init__()

    def forward(self, input, targets):
        # 获取每个批次的大小 N
        N = targets.size()[0]
        # 平滑变量
        smooth = 1
        # 将宽高 reshape 到同一纬度
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        # 计算交集
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        # 计算一个批次中平均每张图的损失
        dice = N_dice_eff.sum() / N
        return dice


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        # 获取每个批次的大小 N
        N = targets.size()[0]
        # 平滑变量
        smooth = 1
        # 将宽高 reshape 到同一纬度
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        # 计算交集
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        # 计算一个批次中平均每张图的损失
        loss = 1 - N_dice_eff.sum() / N
        return loss


class MultiClassDiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(MultiClassDiceLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.kwargs = kwargs

    def forward(self, input, target):
        """
            input tesor of shape = (N, C, H, W,D)
            target tensor of shape = (N, H, W,D)
        """
        # 先将 target 进行 one-hot 处理，转换为 (N, C, H, W,D)
        nclass = input.shape[1]
        target = F.one_hot(target.long(), nclass)
        target = target.permute(0, 4, 1, 2, 3).contiguous()

        assert input.shape == target.shape, "predict & target shape do not match"

        binaryDiceLoss = BinaryDiceLoss()
        total_loss = 0

        # 归一化输出
        logits = F.softmax(input, dim=1)
        C = target.shape[1]

        # 遍历 channel，得到每个类别的二分类 DiceLoss
        for i in range(C):
            dice_loss = binaryDiceLoss(logits[:, i], target[:, i])
            total_loss += dice_loss

        # 每个类别的平均 dice_loss
        return total_loss / C


class Criterion(nn.Module):
    bce_loss_function = nn.CrossEntropyLoss()
    dice_loss_function = MultiClassDiceLoss()

    def __init__(self, weight=None, size_average=True):
        super(Criterion, self).__init__()

    def bce_loss(self, logits, targets):
        loss = self.bce_loss_function(logits, targets)
        return loss

    def dice_loss(self, logits, targets):
        loss = self.dice_loss_function(logits, targets)
        return loss

    def forward(self, logits, targets):
        dice_loss = self.dice_loss(logits, targets)
        bce_loss = self.bce_loss(logits, targets)
        loss = bce_loss * 0.2 + dice_loss * 0.8
        return loss

