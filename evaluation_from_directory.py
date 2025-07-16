import os
import pdb

import numpy as np
import glob
from utils import dice_ratio, ModHausdorffDist

from config import get_args
import csv

from utils.test import get_ASSD, get_RMSD, get_percentile_distance, get_sens, get_spec, get_hd

import nibabel as nib

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
print(LABEL_DIR)
##############################################################################
## Function global parameter
dice_list = []  # dice list
msd_list = []  # mean surface distance list
root_mssd_list = []  # root mean square surface distance
distance_95_list = []
distance_99_list = []
hd_list = []
sens_list = []
spec_list = []


################################################################################
# Functions
################################################################################

def Evaluate(label_dir, pred_dir, pred_id):
    """

    :param label_dir:
    :param pred_dir:
    :param pred_id: string
    :return:
    """
    print('Perform evaluation for subject-%s:' % pred_id)

    print('Loading label...')
    label_file = os.path.join(label_dir, '%s_brain_mask_MNI.nii.gz' % pred_id)

    if not os.path.isfile(label_file):
        print('Plz generate the label file.')
        return

    label = nib.load(label_file)
    label = label.get_data()

    # label = np.load(label_file)
    print('Check label: ', label.shape, np.max(label))

    print('Loading predition...')
    pred_file = os.path.join(pred_dir,
                             '%s_predict_I_brain.nii.gz' % (pred_id))

    if not os.path.isfile(pred_file):
        print('Plz generate the prediction results.')
        return

    pred = nib.load(pred_file)
    pred = pred.get_data()
    print('Check pred: ', pred.shape, np.max(pred))

    print('Extract pred and label for each class...')
    # label_one_hot = one_hot(label)
    # pred_one_hot = one_hot(pred)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    csf_pred = pred.astype(np.uint8)
    csf_label = label.astype(np.uint8)

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

    hd = get_hd(csf_pred, csf_label)
    hd_list.append(hd)
    print("\thd --->{}".format(hd))

    sens = get_sens(csf_pred, csf_label)
    sens_list.append(sens)
    print("\tsens --->{}".format(sens))

    spec = get_spec(csf_pred, csf_label)
    spec_list.append(spec)
    print("\tspec --->{}".format(spec))

    # exit(0)

    # serial results into csv file
    csv_file_path = os.path.join(args.result_dir, 'results_stat.csv')
    # create file
    if not os.path.isfile(csv_file_path):
        f = open(csv_file_path, "w+")
        writer = csv.writer(f)
        # write column head
        # writer.writerow(["scan ID", "CSF DR", "GM DR", "WM DR", "AVG"])
        writer.writerow(
            ["scan ID", "Dice", "Mean Surface Distance", "Root Mean Square Distance", "95 Percentile Distance",
             "99 Percentile Distance", "HD", "Sens", "Spec"])
        f.close()
    # append to file
    with open(csv_file_path, "a+") as f:
        writer = csv.writer(f)
        # write one row
        writer.writerow([str(pred_id),
                         str(csf_dr),
                         str(msd),
                         str(root_mssd),
                         str(distance_95),
                         str(distance_99),
                         str(hd),
                         str(sens),
                         str(spec)
                         ])
        # writer.writerow([str(pred_id),
        #                  str(0),
        #                  str(0),
        #                  str(0),
        #                  str(distance_95),
        #                  str(distance_99),
        #                  str(0),
        #                  str(0),
        #                  str(0)
        #                  ])
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
    glob_pattern = os.path.join(test_label_dir, '*_brain_mask_MNI.nii.gz')
    list_of_test_label_files = glob.glob(glob_pattern)

    # pdb.set_trace()

    for paths in list_of_test_label_files:
        # get file base name
        file_basename = os.path.basename(paths)
        # get id
        file_id = file_basename.replace('_brain_mask_MNI.nii.gz', '')
        # append file id into the list
        list_test_ids.append(file_id)

    return list_test_ids


def get_evl(method_name, paramter_list):
    mean = np.mean(paramter_list)
    std = np.std(paramter_list)
    max = np.max(paramter_list)
    min = np.min(paramter_list)
    # serial results into csv file
    csv_file_path = os.path.join(args.result_dir, 'results_summary.csv')
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
            pred_id=pred_id_single)
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

    mean, std, max_data, min_data = get_evl("HD", hd_list)
    print("{}\t{}\t{}\t{}\t{}".format("HD                  ", mean, std, max_data, min_data))
    mean, std, max_data, min_data = get_evl("Sens", sens_list)
    print("{}\t{}\t{}\t{}\t{}".format("Sens                  ", mean, std, max_data, min_data))
    mean, std, max_data, min_data = get_evl("Spec", spec_list)
    print("{}\t{}\t{}\t{}\t{}".format("Spec                  ", mean, std, max_data, min_data))

    ####################################################################
