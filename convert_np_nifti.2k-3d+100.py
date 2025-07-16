"""
convert for the 2k dataset
"""
import nibabel as nib
import numpy as np
import os
from config import get_args
import shutil
import glob
import pdb


def convert(original_nifty_file,
            numpy_file_name,
            destination_file_name):
    """
    Convert numpy array to nifty format
    :param original_nifty_file:
        template file, from which header and \
        affine function will be extracted
    :param numpy_file_name:
        numpy array will be converted
    :param destination_file_name:
        the converted nifty file
    :return:
    """

    # load original nifty file
    img_org = nib.load(original_nifty_file)
    # get numpy array
    np_arr = np.load(numpy_file_name)
    # create nifty image
    new_img = nib.Nifti1Image(dataobj=np_arr,
                              affine=img_org.affine,
                              header=img_org.header)
    # save image
    nib.save(img=new_img,
             filename=destination_file_name)

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

    # set parameters
    args = get_args()
    checkpoint = args.checkpoint_num
    pre_result_folder = args.result_dir
    raw_top_folder = args.raw_data_dir
    # path to test test_data encoded with hdf5 format
    test_hdf5_file_path = args.test_data_dir
    # pre_folder = os.path.join(pre_result_folder, 'dataset_1k')
    pre_folder = os.path.join(pre_result_folder, 'segmented')
    if not os.path.exists(pre_folder):
        os.makedirs(pre_folder)

    # get list of all test ids
    pattern='*-test-overlap-%d-patch-%d.hdf5' % (args.overlap_step, args.patch_size)
    test_list = get_case_id_list(test_hdf5_file_path, pattern)

    for test_id in test_list:
        # remove prefix 'instance-'
        test_id = test_id.replace('instance-', '')

        # raw_nifty_file_t1c = os.path.join(raw_top_folder, test_id,
        #                                   test_id + '_t1ce.nii.gz')
        # raw_nifty_file_t1n = os.path.join(raw_top_folder, test_id,
        #                                   test_id + '_t1.nii.gz')
        # raw_nifty_file_t2f = os.path.join(raw_top_folder, test_id,
        #                                   test_id + '_flair.nii.gz')
        # raw_nifty_file_t2w = os.path.join(raw_top_folder, test_id,
        #                                   test_id + '_t2.nii.gz')

        raw_nifty_file = os.path.join(raw_top_folder, test_id + '.nii.gz')
        mask_top_folder = raw_top_folder.replace('imagesTr','labelsTr')
        raw_brain_mask_file = os.path.join(mask_top_folder, test_id + '.nii.gz')


        np_arr_file = os.path.join(pre_result_folder,
                                   'test_instance-%s_checkpoint_%d.npy' \
                                   % (test_id, checkpoint))
        # create predict file name by test_id
        dst_file_name = os.path.join(pre_folder, '%s_predict_I_brain.nii.gz' % test_id)

        if not os.path.exists(np_arr_file):
            print(np_arr_file + ' does not exist. Skip.')
            continue

        # convert
        if os.path.exists(dst_file_name):
            print(dst_file_name + ' exists. Skip')
            continue

        # convert(raw_nifty_file, np_arr_file, dst_file_name)
        convert(raw_brain_mask_file,np_arr_file,dst_file_name)
        print(dst_file_name + ' copied.')


        # copy raw nifty file
        # if already copied, then skip
        # raw_brain_mask_file = os.path.join(raw_top_folder,
        #                                    test_id + '_mask.nii.gz')
        # raw_brain_mask_refined_file = os.path.join(raw_top_folder,
        #                                            test_id + '_brain_mask_bin_refined_MNI.nii.gz')
        # raw_brain_mask_file = os.path.join(raw_top_folder,
        #                                    test_id + '_brain_mask_bin_MNI.nii.gz')
        # raw_brain_mask_refined_file = os.path.join(raw_top_folder,
        #                                            test_id + '_brain_mask_bin_refined_MNI.nii.gz')

        # copy files
        shutil.copy2(raw_brain_mask_file, pre_folder)
        # fixed :raw_brain_mask_refined_file is what?
        # shutil.copy2(raw_brain_mask_refined_file, pre_folder)
        shutil.copy2(raw_nifty_file, pre_folder)


        print('Instance %s converted.' % test_id)

    print('All Done!')
