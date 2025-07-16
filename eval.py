import os
import pdb
import numpy as np
import glob
import torch
from config import get_args
import csv
from utils.metric import dice, hd, get_percentile_distance


################################################################################
# Arguments
################################################################################
args = get_args()

LABEL_DIR = args.test_label_dir
PRED_DIR = args.result_dir
PATCH_SIZE = args.patch_size
CHECKPOINT_NUM = args.checkpoint_num  # 153000
OVERLAP_STEPSIZE = args.overlap_step


dice_list = []  # dice list
hd95_list = []  # hd95 list



# Functions
################################################################################
def process_label(label):
    ncr = label == 1
    ed = label == 2
    et = label == 3
    ET = et
    TC = ncr + et
    WT = ncr + et + ed
    return ET, TC, WT

# def overlap_labels(target_tensor):
#     """
#     receives discrete targets and returns overlap ones
#     :param target_tensor: shape (B, C, H, W, D), one-hot-encoded label maps with discrete BraTS labels (ET, NET/NCR, ED)
#     :return: overlapped label maps (ET, TC, WT)
#     """
#     bg = target_tensor[:, 0:1, ...]
#     # necrotic and non-enhancing tumor core
#     net = target_tensor[:, 1:2, ...]
#     # peritumoral edema
#     ed = target_tensor[:, 2:3, ...]
#     # GD-enhancing tumor
#     et = target_tensor[:, 3:4, ...]
#
#     tc = et + net
#     wt = et + net + ed
#     targets = torch.cat([bg, et, tc, wt], dim=1)
#     return targets

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
    # # 将值为4的标签改为3 ----BraTS2020
    label[label == 4] = 3
    # label = label.astype(np.int64)

    print('Check label: ', label.shape, np.max(label))

    print('Loading predition...')
    pred_file = os.path.join(pred_dir,
                             'test_instance-%s_checkpoint_%d.npy' % (pred_id, checkpoint_num))

    if not os.path.isfile(pred_file):
        print('Plz generate the prediction results.')
        return

    pred = np.load(pred_file)

    label_et, label_tc, label_wt = process_label(label)
    infer_et, infer_tc, infer_wt = process_label(pred)
    dice_et = dice(infer_et, label_et)
    dice_tc = dice(infer_tc, label_tc)
    dice_wt = dice(infer_wt, label_wt)

    hd95_et = hd(infer_et,label_et)
    hd95_tc = hd(infer_tc, label_tc)
    hd95_wt = hd(infer_wt, label_wt)

    # hd95_et = get_percentile_distance(infer_et,label_et)
    # hd95_tc = get_percentile_distance(infer_tc,label_tc)
    # hd95_wt = get_percentile_distance(infer_wt,label_wt)


    avg_dice = (dice_et + dice_tc + dice_wt) / 3
    avg_hd = (hd95_et + hd95_tc + hd95_wt) / 3
    # avg_metric = {'avg_dice': avg_dice, 'avg_hd': avg_hd}

    print('Check pred: ', pred.shape, np.max(pred))

    print('Extract pred and label for each class...')

    # evaluate dice ratio
    # csf_dr = dice_ratio(csf_pred, csf_label)
    dice_list.append(avg_dice)
    print("\tDice --->{}".format(avg_dice))
    hd95_list.append(avg_hd)
    print("\tHd95 --->{}".format(avg_hd))



    # serial results into csv file
    csv_file_path = os.path.join(args.result_dir, 'results_stat_' +
                                 str(args.checkpoint_num) + '.csv')
    # create file
    if not os.path.isfile(csv_file_path):
        f = open(csv_file_path, "w+")
        writer = csv.writer(f)
        # write column head
        writer.writerow(
            ["scan ID", "Dice", "95 Percentile Distance"])
        f.close()
    # append to file
    with open(csv_file_path, "a+") as f:
        writer = csv.writer(f)
        writer.writerow([str(pred_id),
                         str(avg_dice),
                         str(avg_hd)
                         ])
        f.close()

    print('Done.')


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


    print("\tmean\tstd\tmax\tmin")
    mean, std, max_data, min_data = get_evl("Dice", dice_list)
    print("{}\t\t\t{}\t{}\t{}\t{}".format("Dice", mean, std, max_data, min_data))

    mean, std, max_data, min_data = get_evl("95 percentile Distance", hd95_list)
    print("{}\t{}\t{}\t{}\t{}".format("95 percentile Distance", mean, std, max_data, min_data))


