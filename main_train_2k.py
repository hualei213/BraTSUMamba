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
from unet.BraTSUmamba_network import BraTSUMamba
from unet.MLP_Vnet import MLP_Vnet
from train_model_2k import train_model
# from train_model_2023_db import train_model
# from train_model_MSD_db import train_model
# from train_model_2023_GLI import train_model
# from train_model_2020 import train_model
# from train_model_M2trans import train_model
from unet.Modified3DUNet import Modified3DUNet
from unet.PrototypeArchitecture3d import PrototypeArchitecture3d
from unet.UNETR import UNETR
from unet.UNeXt_3D import UNeXt3D
from unet.UNeXt_3D_mlp_attention_Parallelfdgmsm_fsca_p import UNeXt_3D_mlp_attention_Parallelfdgmsm_fsca_pc
from unet.UNeXt_3D_mlp_attention_fd_gmsm_fsca_pc import UNeXt_3D_mlp_attention_fd_gmsm_fsca_pc
# from unet import Mlp_UNet3D
# from unet.as_MLP.asmlp import Med_ASMLP
# from unet.unet_mlp import unet_mlp
# from unet.UNeXt_3D_mlp_attention_test import UNeXt3D_mlp_attention
# from unet.UNeXt_3D_mlp_attention_seg_counting_reweighting import UNeXt3D_mlp_attention
# from unet.UNeXt_3D_mlp_attention_boundary import UNeXt3D_mlp_attention_boundary
# from unet.UNeXt_3D_mlp_attention_seg_counting_mamba_multicov import UNeXt3D_mlp_attention
from unet.UNeXt_3D_mlp_attention_seg_counting_mamba import UNeXt3D_mlp_attention
from unet.UNeXt_3D_mlp_attention_seg_counting import UNeXt3D_mlp_attention
from unet.UNeXt_3D_mlp_attention_segmamba_6x import UNeXt_3D_mlp_attention_segmamba_6x
from unet.UNeXt_3D_mlp_attention_fsca import UNeXt3D_mlp_attention_fsca
from unet.UNeXt_3D_mlp_attention_seg_counting_reweighting_mamba import UNeXt3D_mlp_attention_reweight
from unet.UNeXt_3D_mlp_attention_seg_counting_egde import UNeXt3D_mlp_attention_edge
from unet.UNeXt_3D_mlp_attention_seg_counting_mamba_cbam import UNeXt3D_mlp_attention_mamba_cbam
from unet.UNeXt_3D_mlp_attention_seg_counting_mamba_4x_cbam_bsblock import UNeXt_3D_mlp_gmsm_fsca_pc
from unet.UNeXt_3D_mlp_attention_gmsm_fd_fsca_pc import UNeXt_3D_mlp_attention_gmsm_fd_fsca_pc
from unet.UNeXt_3D_mlp_attention_seg_counting_MSBMamba_GAB import UNeXt_3D_mlp_attention_seg_counting_MSBMamba_GAB
from S2CA_net.models.networks import UNet_LGSM_MCA_MSADC_Plus
from SDV_TUnet.model.vision_transformer import SDVTUNet as ViT_seg
from M2FTrans.models import rfnet, mmformer, fusiontrans
from enformer.eoformer import EoFormer
from UNETR.unetr import UNETR
from unet.gmsm_4x import gmsm_4x
from unet.gmsm_4x_pc import gmsm_4x_pc
from unet.gmsm_4x_pc_fusion_RE_ import gmsm_4x_pc_fusion_RE
from unet.gmsm_4x_pc_fusion_re import gmsm_4x_pc_fusion_re
from unet.gmsm_4x_pc_h import gmsm_4x_pc_h
from unet.gmsm_4x_pc_hlcrossatt import gmsm_4x_pc_hlcrossatt
from unet.gmsm_4x_pc_l import gmsm_4x_pc_l
from unet.gmsm_4x_scaf_pc import gmsm_4x_scaf_pc
from unet.gmsm_4x_scaf_pc_fdfusion import gmsm_4x_scaf_pc_fdfusion
from unet.gmsm_4x_scaf_pc_fusion import gmsm_4x_scaf_pc_fusion
from unet.gmsm_4x_scaf_pc_fusion_defreq import gmsm_4x_scaf_pc_fusion_defreq
from unet.gmsm_4x_scaf_pc_fusion_fdreweight import gmsm_4x_scaf_pc_fusion_fdreweight
from unet.msc_2x_mamba import msc_2x_mamba
from unetr_pp.unetr_pp_tumor import UNETR_PP
from tri_segmamba.segmamba import SegMamba

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
    # if args.model_name=="UNeXt_3D_mlp_attention":
    #     model = UNeXt3D_mlp_attention(n_channels=4, n_classes=4)
    if args.model_name=="UNeXt_3D_mlp_attention_pc":
        model = UNeXt3D_mlp_attention(n_channels=4, n_classes=4)
    if args.model_name=="UNeXt_3D_mlp_attention_segmamba_6x":
        model = UNeXt_3D_mlp_attention_segmamba_6x(n_channels=4, n_classes=4)
    if args.model_name=="UNeXt3D_mlp_attention_fsca":
        model = UNeXt3D_mlp_attention_fsca(n_channels=4, n_classes=4)
    if args.model_name=="UNeXt3D_mlp_attention_reweight":
        model = UNeXt3D_mlp_attention_reweight(n_channels=1, n_classes=4)
    if args.model_name=="UNeXt3D_mlp_attention_edge":
        model = UNeXt3D_mlp_attention_edge(n_channels=4, n_classes=4)
    if args.model_name=="UNeXt3D_mlp_attention_mamba_cbam":
        model = UNeXt3D_mlp_attention_mamba_cbam(n_channels=4, n_classes=4)
    if args.model_name=="UNeXt_3D_mlp_gmsm_fsca_pc":
        model = UNeXt_3D_mlp_gmsm_fsca_pc(n_channels=4, n_classes=4)
    if args.model_name=="UNeXt_3D_mlp_attention_gmsm_fd_fsca_pc":
        model = UNeXt_3D_mlp_attention_gmsm_fd_fsca_pc(n_channels=4, n_classes=4)
    if args.model_name=="UNeXt_3D_mlp_attention_fd_gmsm_fsca_pc":
        model = UNeXt_3D_mlp_attention_fd_gmsm_fsca_pc(n_channels=4, n_classes=4)
    if args.model_name=="UNeXt_3D_mlp_attention_Parallelfdgmsm_fsca_pc":
        model = UNeXt_3D_mlp_attention_Parallelfdgmsm_fsca_pc(n_channels=4, n_classes=4)
    if args.model_name=="gmsm_4x_scaf_pc":
        model = gmsm_4x_scaf_pc(n_channels=4, n_classes=4)
    if args.model_name=="gmsm_4x_scaf_pc_fusion":
        model = gmsm_4x_scaf_pc_fusion(n_channels=4, n_classes=4)
    if args.model_name=="gmsm_4x_scaf_pc_fdfusion":
        model = gmsm_4x_scaf_pc_fdfusion(n_channels=4, n_classes=4)
    if args.model_name=="gmsm_4x_scaf_pc_fusion_fdreweight":
        model = gmsm_4x_scaf_pc_fusion_fdreweight(n_channels=4, n_classes=4)
    if args.model_name=="gmsm_4x_scaf_pc_fusion_defreq":
        model = gmsm_4x_scaf_pc_fusion_defreq(n_channels=4, n_classes=4)
    if args.model_name == "gmsm_4x_pc_fusion_re":
        model = gmsm_4x_pc_fusion_re(n_channels=4, n_classes=4)
    if args.model_name == "gmsm_4x_pc_fusion_RE":
        model = gmsm_4x_pc_fusion_RE(n_channels=4, n_classes=4)
    if args.model_name == "gmsm_4x_pc":
        model = gmsm_4x_pc(n_channels=4, n_classes=4)
    if args.model_name == "gmsm_4x_pc_hlcrossatt" or args.model_name=="gmsm_4x_hlcrossatt":
        model = gmsm_4x_pc_hlcrossatt(n_channels=4, n_classes=4)
    if args.model_name == "BraTSUMamba":
        model = BraTSUMamba(n_channels=4, n_classes=4)
    if args.model_name == "gmsm_4x_pc_h":
        model = gmsm_4x_pc_h(n_channels=4, n_classes=4)
    if args.model_name == "gmsm_4x_pc_l":
        model = gmsm_4x_pc_l(n_channels=4, n_classes=4)
    if args.model_name == "gmsm_4x":
        model = gmsm_4x(n_channels=4, n_classes=4)
    if args.model_name=="msc_2x_mamaba":
        model = msc_2x_mamba(n_channels=4, n_classes=4)
    if args.model_name=="UNeXt_3D_mlp_attention_seg_counting_MSBMamba_GAB":
        model = UNeXt_3D_mlp_attention_seg_counting_MSBMamba_GAB(n_channels=4, n_classes=4)
    # if args.model_name=="UNeXt_counting_mamba_6x_cpam":
    #     model = UNeXt_counting_mamba_6x_cpam(n_channels=4, n_classes=4)
    # if args.model_name=="UNeXt_3D_mlp_attention_boundary":
    #     model = UNeXt3D_mlp_attention_boundary(n_channels=4, n_classes=4,deep_flag=False)
    # if args.model_name=="UNeXt_3D_mlp_attention_boundary":
    #     model = UNeXt3D_mlp_attention_boundary(n_channels=4, n_classes=4,deep_flag=False)
    if args.model_name == "s2ca_net":
        model = UNet_LGSM_MCA_MSADC_Plus(num_classes=4, input_channels=4, use_deconv=False, channels=(16, 32, 64, 128),strides=(1, 2, 2, 2), leaky=True, norm='INSTANCE').to(device)
    if args.model_name == "sdv_tunet":
        model = ViT_seg(num_classes=4,
                      embed_dim=96,
                      win_size=7)
    if args.model_name == "m2ftrans":
        model = fusiontrans.Model(num_cls=4)
    if args.model_name == "enformer":
        model = EoFormer(in_channels=4, out_channels=4, drop_path=0.1)
    if args.model_name == "unetr":
        model = UNETR(
            in_channels=4,
            out_channels=4,
            img_size=(128, 128, 128),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed='perceptron',
            norm_name='instance',
            conv_block=True,
            res_block=True,
            dropout_rate=0.0)
    if args.model_name == "unetr++":
        model = UNETR_PP(in_channels=4,
                         out_channels=4,
                         feature_size=16,
                         num_heads=4,
                         depths=[3, 3, 3, 3],
                         dims=[32, 64, 128, 256],
                         do_ds=False,
                         )
    if args.model_name == "segmamba":
        model = SegMamba(in_chans=4, out_chans=4, depths=[2, 2, 2, 2], feat_size=[16, 32, 64, 128])


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