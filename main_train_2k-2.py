import pdb
import random

import numpy as np
import yaml

from Criterion import Criterion
from config import get_args
import os
import sys
import torch.nn as nn
import torch.cuda
import torch.optim.lr_scheduler as lr_scheduler
from torch import optim

from unet import UNet3D, UNeXt3D_ablation
from unet.MLP_Vnet import MLP_Vnet
# from train_parallel import train_model, test_model
from train_model_2k import train_model
from unet.Modified3DUNet import Modified3DUNet
from unet.PrototypeArchitecture3d import PrototypeArchitecture3d
from unet.UNETR import UNETR
from unet.UNeXt_3D import UNeXt3D
# from unet import Mlp_UNet3D
# from unet.as_MLP.asmlp import Med_ASMLP
from unet.unet_mlp import unet_mlp
# from unet.UNeXt_3D_mlp_attention_test import UNeXt3D_mlp_attention
from unet.UNeXt_3D_mlp_attention_seg_counting import UNeXt3D_mlp_attention
from unet.UNeXt_3D_mlp_attention_boundary import UNeXt3D_mlp_attention_boundary


def setup_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # get option
    args = get_args()
    # set random seed
    setup_seed(args.random_seed)

    # set GPUs
    # import torch

    # print(torch.cuda.get_device_name(0))  # 检查第一个 GPU

    if torch.cuda.device_count() >= 1:
        print("Let's use GPUs: " + args.gpus)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    '''
         # intialize model
    '''
    if args.model_name == "UNETR":
        model =UNETR(
            in_channels=1,
            out_channels=args.class_num,
            img_size=(64, 64, 64),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            conv_block=False,
            res_block=True,
            dropout_rate=0.0,
        )

    if args.model_name == "UNeXt3D":
        model = UNeXt3D(n_channels=1, n_classes=args.class_num)#6
    if args.model_name == "UNeXt3D_ablation":
        model = UNeXt3D_ablation(n_channels=1, n_classes=args.class_num)
    if args.model_name == "EDNet":
        model = PrototypeArchitecture3d(n_channels=1, n_classes=args.class_num)
    if args.model_name == "UNet3D":
        model = UNet3D(n_channels=1, n_classes=args.class_num)
    if args.model_name == "ModifyUnet":
        model = Modified3DUNet(n_channels=1, n_classes=args.class_num)
    if args.model_name == "UNet3D":
        model = UNet3D(n_channels=1, n_classes=args.class_num)
    # if args.model_name=="asmlp":
    #     model = Med_ASMLP()
    if args.model_name=="mlp_vnet":
        model = MLP_Vnet()
    if args.model_name=="UNeXt_3D_mlp_attention":
        model = UNeXt3D_mlp_attention(n_channels=4, n_classes=4)
    if args.model_name=="UNeXt_3D_mlp_attention_boundary":
        model = UNeXt3D_mlp_attention_boundary(n_channels=4, n_classes=4,deep_flag=False)
    # multiple GPUs
    # if torch.cuda.is_available():
    #     num_GPU = torch.cuda.device_count()
    #     model = nn.DataParallel(model, device_ids=[int(item) for item in args.gpus.split(',')])

    model.to(device)

    ###############################################################################
    # Train the model
    if args.mode == 'train':
        ############################################################################
        # set optimization parameters
        # criterion = nn.CrossEntropyLoss()
        criterion = Criterion()

        # Observe that all parameters are being optimized
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=2e-6
                               )
        #registering optimizer
        # optimizer_reg = torch.optim.Adam(model.parameters(), lr=args.lr_reg)

        # prepare registering loss
        # if args.image_loss == 'ncc':
        #     image_loss_func = vxm_losses.NCC().loss
        # elif args.image_loss == 'mse':
        #     image_loss_func = vxm_losses.MSE().loss
        # else:
        #     raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

        # if args.bidir:
        #     losses_reg = [image_loss_func, image_loss_func]
        #     weights_reg = [0.5, 0.5]
        # else:
        #     losses_reg = [image_loss_func]
        #     weights_reg = [1]

        # # prepare deformation loss
        # losses_reg += [vxm_losses.Grad('l2', loss_mult=args.int_downsize).loss]
        # weights_reg += [args.weight_reg]



        # Decay LR by a factor of 0.1 every 5000 epochs
        #500000
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer,
                                               step_size=500,
                                               gamma=0.1)
        # exp_lr_scheduler=lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100,factor=0.01,verbose=True)

        ############################################################################
        # train
        try:
            os.makedirs(args.log_dir,exist_ok=True)
            with open(args.log_dir+"config.yml","w") as f:
                yaml.dump(args,f)
            train_model(model=model,
                        criterion=criterion,
                        optimizer=optimizer,
                        scheduler=exp_lr_scheduler,
                        num_epochs=args.epochs,
                        resume_train=args.resume,
                        epochs_per_val=args.epochs_per_vali,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        result_dir=args.result_dir,
                        device=device,
                        log_dir=args.log_dir
                        )
        except KeyboardInterrupt:
            torch.save(model.state_dict(), 'INTERRUPTED.pth')
            print('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                exit(0)

    ################################################################################
    else:
        print('Error. Mode should be train')
        sys.exit(0)