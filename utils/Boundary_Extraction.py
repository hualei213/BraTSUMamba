import torch
import numpy as np
from torch import einsum
from torch import Tensor, nn
from scipy.ndimage import distance_transform_edt as distance
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union


def get_sobel(in_chan, out_chan):
    # https://cloud.tencent.com/developer/ask/sof/100045389
    filter_x = np.array([
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    ]).astype(np.float32)
    filter_y = np.array([
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    ]).astype(np.float32)
    filter_z = np.array([
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
    ]).astype(np.float32)

    filter_x = filter_x.reshape((1, 1, 3, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_z = filter_z.reshape((1, 1, 3, 3, 3))
    filter_z = np.repeat(filter_z, in_chan, axis=1)
    filter_z = np.repeat(filter_z, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_z = torch.from_numpy(filter_z)

    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    filter_z = nn.Parameter(filter_z, requires_grad=False)

    conv_x = nn.Conv3d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv3d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    conv_z = nn.Conv3d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_z.weight = filter_z

    sobel_x = nn.Sequential(conv_x, nn.BatchNorm3d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm3d(out_chan))
    sobel_z = nn.Sequential(conv_z, nn.BatchNorm3d(out_chan))

    if torch.cuda.device_count() >= 1:
        device = torch.device("cuda")
        # device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # return sobel_x.to(device), sobel_y.to(device), sobel_z.to(device)
    return sobel_x, sobel_y, sobel_z


def run_sobel(conv_x, conv_y, conv_z, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g_z = conv_z(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2) + torch.pow(g_z, 2))
    # g = torch.sigmoid(g)
    return torch.sigmoid(g) * input


def run_sobel2gt(conv_x, conv_y, conv_z, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g_z = conv_z(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2) + torch.pow(g_z, 2))
    return torch.where(g > 0.5, 1., 0.)
