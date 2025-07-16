# from config import get_args
import os
import random
import sys

import numpy as np
import torch.nn as nn
import torch.cuda
import glob
import pdb



from config import get_args
from unet import UNet3D, UNeXt3D_ablation
from unet.Modified3DUNet import Modified3DUNet
from unet.PrototypeArchitecture3d import PrototypeArchitecture3d
from unet.UNETR import UNETR
from unet.UNeXt_3D import UNeXt3D
from test_model_2k import test_model
# from unet.UNeXt_3D_mlp_attention_test import UNeXt3D_mlp_attention
from unet.UNeXt_3D_mlp_attention_seg_counting import UNeXt3D_mlp_attention
from unet.UNeXt_3D_mlp_attention_boundary import UNeXt3D_mlp_attention_boundary
from unet.unet_mlp import unet_mlp
from unet.MLP_Vnet import MLP_Vnet


def setup_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_case_id(case_name, pattern):
    """
    get case id, based on case file name
    :param case_name: file name of case
    :return: case id
    """
    # pattern = '_raw_t1_MNI.nii.gz'

    # pdb.set_trace()

    case_id = case_name.replace(pattern, '')

    return case_id


def get_case_id_list(str_top_path, pattern):
    """
    get ids of call cases
    :param str_top_path: top path contains all cases
    :return: list of all case ids
    """

    # pdb.set_trace()
    # set parameters
    all_case_id = []

    # set pattern
    # pattern = '*_raw_t1_MNI.nii.gz'

    os.chdir(str_top_path)

    # get file names of all cases matching the pattern
    case_name_list = glob.glob(pattern)

    for i in range(len(case_name_list)):
        case_name = case_name_list[i]
        case_id = get_case_id(case_name, pattern.replace('*',""))
        all_case_id.append(case_id)

    return all_case_id


if __name__ == '__main__':
    # torch.backends.cudnn.enabled = False
    # get option
    args = get_args()
    # set random seed
    setup_seed(args.random_seed)
    # set parameters
    test_hdf5_file_path = args.test_data_dir

    # set GPUs
    if torch.cuda.device_count() >= 1:
        print("Let's use GPUs: " + args.gpus)
        # device = torch.device("cuda: " + args.gpus)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # intialize model
    if args.model_name =="UNETR":
        # model = UNet3D(n_channels=1, n_classes=args.class_num)
        # model = PrototypeArchitecture3d(n_channels=1, n_classes=args.class_num)
        model = UNETR(
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
        # model = Modified3DUNet(n_channels=1, n_classes=args.class_num)

    if args.model_name == "UNeXt3D":
        model = UNeXt3D(n_channels=1, n_classes=args.class_num)
    if args.model_name == "UNeXt3D_ablation":
        model = UNeXt3D_ablation(n_channels=1, n_classes=args.class_num)
    if args.model_name == "EDNet":
        model = PrototypeArchitecture3d(n_channels=1, n_classes=args.class_num)
    if args.model_name == "UNet3D":
        model = UNet3D(n_channels=1, n_classes=args.class_num)
    if args.model_name == "ModifyUnet":
        model = Modified3DUNet(n_channels=1, n_classes=args.class_num)
    if args.model_name=="mlp_vnet":
        model = MLP_Vnet()
    if args.model_name=="UNeXt_3D_mlp_attention":
        model = UNeXt3D_mlp_attention(n_channels=4, n_classes=4)
    if args.model_name=="UNeXt_3D_mlp_attention_boundary":
        model = UNeXt3D_mlp_attention_boundary(n_channels=4, n_classes=4,deep_flag=False)
    # multiple GPUs

    #if torch.cuda.is_available():
        #num_GPU = torch.cuda.device_count()
        #model = nn.DataParallel(model, device_ids=[int(item) for item in args.gpus.split(',')])

    model.to(device)

    ##################################################################################
    # test the model
    if args.mode == 'predict':

        # load saved model
        saved_model_file = os.path.join(args.model_dir,
                                        ('model-%d') % args.checkpoint_num)
        model.load_state_dict(torch.load(saved_model_file,map_location='cuda:0'))
        print('Model loaded from {}'.format(saved_model_file))

        model.eval()

        # get list of all test ids
        pattern = '*-test-overlap-%d-patch-%d.hdf5' %(args.overlap_step,args.patch_size)
        test_list = get_case_id_list(test_hdf5_file_path, pattern)

        for test_instance_id in test_list:
            print('-' * 20 + 'predict instance %s' % test_instance_id + '-' * 20)
            test_model(model=model,
                       device=device,
                       test_instance_id=test_instance_id)

        print('All done.')
        sys.exit(0)

    else:
        print('Error. Only support model prediction.')
        sys.exit(0)
