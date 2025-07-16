import time

import numpy as np
import torch
import torchvision
# from pytorch_model_summary import summary
from thop import profile
# from torchstat import stat
import unet
from unet import *
# from ptflops import get_model_complexity_info

# from unet import Mlp_UNet3D
from unet.Modified3DUNet import Modified3DUNet
from unet.PrototypeArchitecture3d import PrototypeArchitecture3d
from unet.UNETR import UNETR

from unet.UNeXt_3D import  UNeXt3D
from unet import UNet3D
from unet.unet3D import _3DUNet
from unet.unet_mlp import unet_mlp


def get_params_flops(model):
    input_data = torch.randn(1, 1, 128, 128, 128)
    # input_data = torch.randn(1, 1, 64, 64, 64)
    flops, params = profile(model, (input_data,))
    print('flops:', flops, 'params:', params)
    print('flops:%.2f G,params: %.2f M' % (flops / 1e9, params / 1e6))

def speeds(model):
    device = torch.device("cuda")
    model.eval()
    model.to(device)
    dummy_input = torch.randn(1, 1, 128, 128, 128, dtype=torch.float).to(device)
    # dummy_input = torch.randn(1, 1, 64, 64, 64, dtype=torch.float).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn,
                                                                                         std_syn=std_syn,
                                                                                         mean_fps=mean_fps))
    print(mean_syn)


if __name__=="__main__":

    # model = Mlp_UNet3D(1, 2)
    # model = UNeXt3D(1, 2)
    # model = UNeXt3D_ablation(1, 2)
    # model = UNet3D(1, 2)
    # model = Modified3DUNet(1,2)
    # model = PrototypeArchitecture3d(1,2)
    # model = UNETR(in_channels=1,out_channels=2,img_size=(64, 64, 64),feature_size=16,hidden_size=768,mlp_dim=3072,num_heads=12,pos_embed="perceptron",norm_name="instance",conv_block=False,res_block=True,dropout_rate=0.0,)
    # model = unet_mlp(1,2)
    # model =net8(1,2)
    model = _3DUNet(in_channels=1, num_classes=2, batch_normal=True, bilinear=False)
    get_params_flops(model)
    # speeds(model)
    # speeds2(model)


"""
Unet-3D
flops: 76977537024.0 params: 2418594.0
flops:76.98 G,params: 2.42 M

UneXt-3D
flops: 1006633984.0 params: 2678646.0
flops:1.01 G,params: 2.68 M

Modified3DUNet
flops: 7182376960.0 params: 1781304.0
flops:7.18 G,params: 1.78 M

PrototypeArchitecture3d
flops: 117621490688.0 params: 34927277.0
flops:117.62 G,params: 34.93 M

UNETR
flops: 21461336064.0 params: 92247778.0
flops:21.46 G,params: 92.25 M
"""