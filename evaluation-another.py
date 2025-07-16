import os
import pdb

import numpy as np
import glob
from utils import dice_ratio, ModHausdorffDist

from config import get_args
import csv

# from utils.test import get_ASSD, get_RMSD, get_percentile_distance, get_hd95, get_sens, get_spec
from utils.test import get_ASSD, get_RMSD, get_percentile_distance, get_sens, get_spec

from monai.transforms import Activations, AsDiscrete, Compose
from monai.metrics import DiceMetric, HausdorffDistanceMetric

from myfile.utils.data_pipeline import InferenceReader, overlap_labels, discretize_labels, ETThresholdSuppression, \
    RemoveMinorConnectedComponents

from myfile.utils.iterator import MetricMeter

"""Perform evaluation in terms of dice ratio and 3D MHD.
"""

################################################################################
# Arguments
################################################################################
args = get_args()

LABEL_DIR = args.test_label_dir
PRED_DIR = args.result_dir
PATCH_SIZE = args.patch_size
CHECKPOINT_NUM = args.checkpoint_num  # 153000
OVERLAP_STEPSIZE = args.overlap_step

# PRED_ID = test_list
# PRED_ID = [501]

##############################################################################
## Function global parameter
dice_list = []  # dice list
msd_list = []  # mean surface distance list
root_mssd_list = []  # root mean square surface distance
distance_95_list = []
distance_99_list = []
hd95_list = []
sens_list = []
spec_list = []


suppress_thr=400
is_global=True


################################################################################
# Functions
################################################################################
post_trans = Compose([Activations(sigmoid=True),
                      AsDiscrete(threshold_values=True),
                      RemoveMinorConnectedComponents(10),
                      ETThresholdSuppression(thr=suppress_thr, is_overlap=True, global_replace=is_global)
                      ])

def overlap_labels(target_tensor):
    """
    receives discrete targets and returns overlap ones
    :param target_tensor: shape (B, C, H, W, D), one-hot-encoded label maps with discrete BraTS labels (ET, NET/NCR, ED)
    :return: overlapped label maps (ET, TC, WT)
    """
    bg = target_tensor[:, 0:1, ...]
    # necrotic and non-enhancing tumor core
    net = target_tensor[:, 1:2, ...]
    # peritumoral edema
    ed = target_tensor[:, 2:3, ...]
    # GD-enhancing tumor
    et = target_tensor[:, 3:4, ...]

    tc = et + net
    wt = et + net + ed
    targets = torch.cat([bg, et, tc, wt], dim=1)
    return targets



def one_hot(label):
    '''Convert label (d,h,w) to one-hot label (d,h,w,num_class).
    '''

    # num_class = np.max(label) + 1
    num_class = 4
    return np.eye(num_class)[label]


def MHD_3D(pred, label):
    '''Compute 3D MHD for a single class.

    Args:
        pred: An array of size [Depth, Height, Width], with only 0 or 1 values
        label: An array of size [Depth, Height, Width], with only 0 or 1 values

    Returns:
        3D MHD for a single class
    '''

    D, H, W = label.shape

    pred_d = np.array([pred[:, i, j] for i in range(H) for j in range(W)])
    pred_h = np.array([pred[i, :, j] for i in range(D) for j in range(W)])
    pred_w = np.array([pred[i, j, :] for i in range(D) for j in range(H)])

    label_d = np.array([label[:, i, j] for i in range(H) for j in range(W)])
    label_h = np.array([label[i, :, j] for i in range(D) for j in range(W)])
    label_w = np.array([label[i, j, :] for i in range(D) for j in range(H)])

    MHD_d = ModHausdorffDist(pred_d, label_d)[0]
    MHD_h = ModHausdorffDist(pred_h, label_h)[0]
    MHD_w = ModHausdorffDist(pred_w, label_w)[0]

    ret = np.mean([MHD_d, MHD_h, MHD_w])

    print('--->MHD d:', MHD_d)
    print('--->MHD h:', MHD_h)
    print('--->MHD w:', MHD_w)
    # print('--->avg:', ret)

    return ret


def Evaluate(label_dir, pred_dir, pred_id, patch_size, checkpoint_num,
             overlap_step):
    """

    :param label_dir:
    :param pred_dir:
    :param pred_id: string
    :param patch_size:
    :param checkpoint_num:
    :param overlap_step:
    :return:
    """
    print('Perform evaluation for subject-%s:' % pred_id)

    print('Loading label...')
    label_file = os.path.join(label_dir, 'instance-%s-label.npy' % pred_id)

    if not os.path.isfile(label_file):
        print('Plz generate the label file.')
        return

    label = np.load(label_file)
    print('Check label: ', label.shape, np.max(label))

    print('Loading predition...')
    pred_file = os.path.join(pred_dir,
                             'test_instance-%s_checkpoint_%d.npy' % (pred_id, checkpoint_num))

    if not os.path.isfile(pred_file):
        print('Plz generate the prediction results.')
        return

    pred = np.load(pred_file)



    pred = post_trans(pred.unsqueeze(0))

    pred = overlap_labels(pred)




    # compute dice and hausdorff distance 95
    dice_metric = DiceMetric(include_background=True, reduction='mean')
    hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction='mean', percentile=95)


    dice_et, dice_et_not_nan = dice_metric(y_pred=pred[:, 1:2, ...], y=label[:, 1:2, ...])
    dice_tc, dice_tc_not_nan = dice_metric(y_pred=pred[:, 2:3, ...], y=label[:, 2:3, ...])
    dice_wt, dice_wt_not_nan = dice_metric(y_pred=pred[:, 3:4, ...], y=label[:, 3:4, ...])

    hd95_et, hd95_et_not_nan = hausdorff_metric(y_pred=pred[:, 1:2, ...],
                                                y=label[:, 1:2, ...])
    hd95_tc, hd95_tc_not_nan = hausdorff_metric(y_pred=pred[:, 2:3, ...],
                                                y=label[:, 2:3, ...])
    hd95_wt, hd95_wt_not_nan = hausdorff_metric(y_pred=pred[:, 3:4, ...],
                                                y=label[:, 3:4, ...])

    # post-process Dice and HD95 values
    # if subject has no enhancing tumor, empty prediction yields Dice of 1 and HD95 of 0
    # otherwise, false positive yields Dice of 0 and HD95 of 373.13 (worst single case)
    if dice_et_not_nan.item() == 0:
        if pred[:, 1:2, ...].max() == 0 and label[:, 1:2, ...].max() == 0:
            dice_et = torch.as_tensor(1)
        else:
            dice_et = torch.as_tensor(0)

    if hd95_et_not_nan.item() == 0:
        if pred[:, 1:2, ...].max() == 0 and label[:, 1:2, ...].max() == 0:
            hd95_et = torch.as_tensor(0)
        else:
            hd95_et = torch.as_tensor(373.13)

    if hd95_et.item() == np.inf:
        hd95_et = torch.as_tensor(373.13)

    et_metric = {'et_dice': dice_et.item(), 'et_hd': hd95_et.item()}
    tc_metric = {'tc_dice': dice_tc.item(), 'tc_hd': hd95_tc.item()}
    wt_metric = {'wt_dice': dice_wt.item(), 'wt_hd': hd95_wt.item()}

    avg_dice = (dice_et.item() + dice_tc.item() + dice_wt.item()) / 3
    avg_hd = (hd95_et.item() + hd95_tc.item() + hd95_wt.item()) / 3
    avg_metric = {'avg_dice': avg_dice, 'avg_hd': avg_hd}



    print('Check pred: ', pred.shape, np.max(pred))

    print('Extract pred and label for each class...')
    # label_one_hot = one_hot(label)
    # pred_one_hot = one_hot(pred)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    csf_pred = pred
    csf_label = label

    # print('Check shape: ', label_one_hot.shape, pred_one_hot.shape)

    # evaluate dice ratio

    csf_dr = dice_ratio(csf_pred, csf_label)
    dice_list.append(csf_dr)
    print("\tDice --->{}".format(csf_dr))
    msd = get_ASSD(csf_pred, csf_label)
    msd_list.append(msd)
    print("\tMsd --->{}".format(msd))

    root_mssd = get_RMSD(csf_pred, csf_label)
    root_mssd_list.append(root_mssd)
    print("\troot_mssd --->{}".format(root_mssd))

    distance_95 = get_percentile_distance(csf_pred, csf_label, percentile=95)
    distance_95_list.append(distance_95)
    print("\tdistance_95 --->{}".format(distance_95))

    distance_99 = get_percentile_distance(csf_pred, csf_label, percentile=99)
    distance_99_list.append(distance_99)
    print("\tdistance_99 --->{}".format(distance_99))

    # hd_95 = get_hd95(csf_pred, csf_label)
    # hd95_list.append(hd_95)
    # print("\thd_95 --->{}".format(hd_95))

    sens = get_sens(csf_pred, csf_label)
    sens_list.append(sens)
    print("\tsens --->{}".format(sens))

    spec = get_spec(csf_pred, csf_label)
    spec_list.append(spec)
    print("\tspec --->{}".format(spec))

    # exit(0)

    # serial results into csv file
    csv_file_path = os.path.join(args.result_dir, 'results_stat_' +
                                 str(args.checkpoint_num) + '.csv')
    # create file
    if not os.path.isfile(csv_file_path):
        f = open(csv_file_path, "w+")
        writer = csv.writer(f)
        # write column head
        # writer.writerow(["scan ID", "CSF DR", "GM DR", "WM DR", "AVG"])
        # writer.writerow(["scan ID", "Dice", "Mean Surface Distance", "Root Mean Square Distance","95 Percentile Distance","99 Percentile Distance","HD95","Sens","Spec"])
        writer.writerow(
            ["scan ID", "Dice", "Mean Surface Distance", "Root Mean Square Distance", "95 Percentile Distance",
             "99 Percentile Distance", "Sens", "Spec"])
        f.close()
    # append to file
    with open(csv_file_path, "a+") as f:
        writer = csv.writer(f)
        # write one row
        # writer.writerow([str(pred_id),
        #                  str(csf_dr),
        #                  str(msd),
        #                  str(root_mssd),
        #                  str(distance_95),
        #                  str(distance_99),
        #                  str(hd_95),
        #                  str(sens),
        #                  str(spec)
        #                  ])
        writer.writerow([str(pred_id),
                         str(csf_dr),
                         str(msd),
                         str(root_mssd),
                         str(distance_95),
                         str(distance_99),
                         str(sens),
                         str(spec)
                         ])
        f.close()

    print('Done.')


def get_test_list(test_label_dir):
    """
    get list of test samples
    :param test_label_dir: directory contains labels of all test instances
    :return: list contains ids of test instances
    """
    ### set parameters
    list_test_ids = []

    # get path of all test label files
    glob_pattern = os.path.join(test_label_dir, 'instance*label.npy')
    list_of_test_label_files = glob.glob(glob_pattern)

    # pdb.set_trace()

    for paths in list_of_test_label_files:
        # get file base name
        file_basename = os.path.basename(paths)
        # get id
        file_id = file_basename.replace('instance-', '').replace('-label.npy', '')
        # append file id into the list
        list_test_ids.append(file_id)

    return list_test_ids


def get_evl(method_name, paramter_list):
    mean = np.mean(paramter_list)
    std = np.std(paramter_list)
    max = np.max(paramter_list)
    min = np.min(paramter_list)
    # serial results into csv file
    csv_file_path = os.path.join(args.result_dir, 'results_summary_' +
                                 str(args.checkpoint_num) + '.csv')
    # create file
    if not os.path.isfile(csv_file_path):
        f = open(csv_file_path, "w+")
        writer = csv.writer(f)
        # write column head
        writer.writerow(["Method", "Mean", "Std", "Max", "Min"])
        f.close()
    # append to file
    with open(csv_file_path, "a+") as f:
        writer = csv.writer(f)
        # write one row
        writer.writerow([str(method_name),
                         str(mean),
                         str(std),
                         str(max),
                         str(min)
                         ])
        f.close()
    return mean, std, max, min


if __name__ == '__main__':
    # get prediction ID
    PRED_ID = get_test_list(LABEL_DIR)
    for pred_id_single in PRED_ID:
        Evaluate(
            label_dir=LABEL_DIR,
            pred_dir=PRED_DIR,
            pred_id=pred_id_single,
            patch_size=PATCH_SIZE,
            checkpoint_num=CHECKPOINT_NUM,
            overlap_step=OVERLAP_STEPSIZE)
    ###################################################################
    print("==============================================")
    print("\tmean\tstd\tmax\tmin")
    mean, std, max_data, min_data = get_evl("Dice", dice_list)
    print("{}\t\t\t{}\t{}\t{}\t{}".format("Dice", mean, std, max_data, min_data))
    mean, std, max_data, min_data = get_evl("Mean Distance", msd_list)
    print("{}\t\t{}\t{}\t{}\t{}".format("Mean Distance", mean, std, max_data, min_data))
    mean, std, max_data, min_data = get_evl("Root Mean Distance", root_mssd_list)
    print("{}\t{}\t{}\t{}\t{}".format("Root Mean Distance", mean, std, max_data, min_data))
    mean, std, max_data, min_data = get_evl("95 percentile Distance", distance_95_list)
    print("{}\t{}\t{}\t{}\t{}".format("95 percentile Distance", mean, std, max_data, min_data))
    mean, std, max_data, min_data = get_evl("99 percentile Distance", distance_99_list)
    print("{}\t{}\t{}\t{}\t{}".format("99 percentile Distance", mean, std, max_data, min_data))

    # mean, std, max_data, min_data = get_evl("HD95", hd95_list)
    # print("{}\t{}\t{}\t{}\t{}".format("HD95                  ", mean, std, max_data, min_data))
    mean, std, max_data, min_data = get_evl("Sens", sens_list)
    print("{}\t{}\t{}\t{}\t{}".format("Sens                  ", mean, std, max_data, min_data))
    mean, std, max_data, min_data = get_evl("Spec", spec_list)
    print("{}\t{}\t{}\t{}\t{}".format("Spec                  ", mean, std, max_data, min_data))

    ####################################################################
