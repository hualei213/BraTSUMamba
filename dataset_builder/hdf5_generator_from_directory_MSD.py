"""
generate train/val/test records from dataset2k
"""
import os
import glob
import random

import nibabel as nib
import numpy as np
import sys
import csv
import math
from random import shuffle
from time import gmtime, strftime

sys.path.append('../')
from config import get_args

from hdf5_generator_MSD import write_training_examples, \
    write_validation_examples, \
    write_test_examples, \
    cut_edge, \
    per_image_normalize

import pdb


def get_case_path_list(str_top_path):
    """
    Get path of cases
    :param str_top_path: path of top directory contains all cases
    :return: list contains path of each raw case
             and list contains path of each REGISTERED raw case
    """

    raw_t1_pattern = '*raw_t1.nii.gz'
    raw_t1_MNI_pattern = '*raw_t1_MNI.nii.gz'

    # list of all raw cases
    raw_t1_list = glob.glob(os.path.join(str_top_path, raw_t1_pattern))
    # list of all registered raw cases
    raw_t1_MNI_list = glob.glob(os.path.join(str_top_path, raw_t1_MNI_pattern))

    return raw_t1_list, raw_t1_MNI_list


def get_case_id_list(str_top_path, pattern):
    """
    get ids of call cases
    :param str_top_path: top path contains all cases
    :return: list of all case ids
    """

    # set parameters
    all_case_id = []

    # set pattern
    # pattern = '*_raw_t1_MNI.nii.gz'

    os.chdir(str_top_path)

    # get file names of all cases matching the pattern
    case_name_list = glob.glob(pattern)

    case_id_pattern = '_raw_t1_MNI.nii.gz'
    for i in range(len(case_name_list)):
        case_name = case_name_list[i]
        case_id = get_case_id(case_name, case_id_pattern)
        all_case_id.append(case_id)

    return all_case_id


def load_subject(raw, brain_mask_path):
    """
    load case related test_data
    :param case_path: path to case
    :param brain_mask_path: path to brain mask
    :return: [T1, label]
    """

    # load brain test_data
    # img_T1c = nib.load(case_path_t1c)
    # img_T1n = nib.load(case_path_t1n)
    # img_T2f = nib.load(case_path_t2f)
    # img_T2w = nib.load(case_path_t2w)
    img_raw = nib.load(raw)



    # inputs_T1c = img_T1c.get_fdata()
    # inputs_T1n = img_T1n.get_fdata()
    # inputs_T2f = img_T2f.get_fdata()
    # inputs_T2w = img_T2w.get_fdata()
    inputs_raw = img_raw.get_fdata()


    # expand inputs_T1 as 4 dimensional, which is required by conv3D @ tensorflow
    # inputs_T1c = np.expand_dims(inputs_T1c, axis=3)
    # inputs_T1n = np.expand_dims(inputs_T1n, axis=3)
    # inputs_T2f = np.expand_dims(inputs_T2f, axis=3)
    # inputs_T2w = np.expand_dims(inputs_T2w, axis=3)
    # inputs_raw = np.expand_dims(inputs_raw,axis=3)

    # get case id
    img_label = nib.load(brain_mask_path)
    inputs_label = img_label.get_fdata()

    # expand dimensions
    inputs_label = np.expand_dims(inputs_label, axis=3)
    # cast to uint8
    inputs_label = inputs_label.astype(np.uint8)

    # pdb.set_trace()

    return inputs_raw, inputs_label


def generate_test_file(case_id,
                       raw_data_path,
                       brain_mask_path,
                       output_path,
                       patch_size,
                       overlap_stepsize):
    """
    generate hdf5 records
    case_id: id of current case
    :param raw_data_path: path to raw test_data file
    brain_mask_path: path to brain mask file
    :param output_path: path to output hdf5 records
    :param patch_size: patch size
    :param overlap_stepsize: overlap step size
    :return:
    """

    # pdb.set_trace()

    # set flag
    file_not_exist_flag = False

    test_subject_name = '%s-test-overlap-%d-patch-%d.hdf5' \
                        % (case_id, patch_size-overlap_stepsize, patch_size)
    test_filename = os.path.join(output_path, test_subject_name)

    if (not os.path.isfile(test_filename)):
        file_not_exist_flag = True

    label_filename = os.path.join(output_path,
                                  '%s-label.npy' % case_id)
    if (not os.path.isfile(label_filename)):
        file_not_exist_flag = True

    ##################################################################################
    # load test_data
    if file_not_exist_flag:
        print('Loading test_data...')
        # load image and label
        [_T1, _label] = load_subject(raw_data_path, brain_mask_path)
        # cut edge
        (original_shape, cut_size) = cut_edge(_T1)
        # normalize image
        _T1 = per_image_normalize(_T1)

        print('Check original_shape: ', original_shape)
        print('Check cut_size: ', cut_size)
    else:
        print('Test file ' + test_subject_name + ' already exist.')
        return

    ##################################################################################
    # write records
    print('Create the test file:')
    write_test_examples(T1=_T1,
                        original_shape=original_shape,
                        patch_size=patch_size,
                        cut_size=cut_size,
                        overlap_stepsize=overlap_stepsize,
                        output_file=test_filename)

    # write label file
    print('Create the converted label file:')  # added
    np.save(label_filename, _label[:, :, :, 0].astype(np.uint8))  # added

    print('---Done.---')


def generate_files(raw_data_dir, output_path,
                   train_id_list, valid_id_list,
                   test_id_list, patch_size, overlap_stepsize):
    """
    Generate hdf5 records
    :param raw_data_dir: directory contains raw test_data
    :param output_path: directory for hdf5 records
    :param train_id_list: list of training instances
    :param valid_id_list: list of validation instances
    :param test_id_list:  list of testing instances
    :param patch_size: patch size
    :param overlap_stepsize: overlap step size
    :return:
    """
    # prepare directories
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    train_records_path = os.path.join(output_path, 'train')
    valid_records_path = os.path.join(output_path, 'val')
    valid_label_path = os.path.join(output_path, 'val_label')
    test_records_path = os.path.join(output_path, 'test')
    test_label_path = os.path.join(output_path, 'test_label')

    if not os.path.exists(train_records_path):
        os.makedirs(train_records_path)
    if not os.path.exists(valid_records_path):
        os.makedirs(valid_records_path)
    if not os.path.exists(test_records_path):
        os.makedirs(test_records_path)
    if not os.path.exists(test_label_path):
        os.makedirs(test_label_path)
    if not os.path.exists(valid_label_path):
        os.makedirs(valid_label_path)

    all_id_list = train_id_list + valid_id_list + test_id_list  # get id list of all subjects
    for case_id in all_id_list:
        print('---Process subject %s:---' % case_id)

        file_not_exist_flag = False
        if case_id in train_id_list:
            train_subject_name = 'instance-%s.hdf5' % case_id
            train_filename = os.path.join(train_records_path, train_subject_name)
            if (not os.path.isfile(train_filename)):
                file_not_exist_flag = True

        if case_id in valid_id_list:
            valid_subject_name = 'instance-%s-valid-overlap-%d-patch-%d.hdf5' \
                                 % (case_id,patch_size- overlap_stepsize, patch_size)
            valid_filename = os.path.join(valid_records_path, valid_subject_name)
            if (not os.path.isfile(valid_filename)):
                file_not_exist_flag = True
            label_filename = os.path.join(valid_label_path,
                                          'instance-%s-label.npy' % case_id)
            if (not os.path.isfile(label_filename)):
                file_not_exist_flag = True
        if case_id in test_id_list:
            test_subject_name = 'instance-%s-test-overlap-%d-patch-%d.hdf5' \
                                % (case_id,patch_size- overlap_stepsize, patch_size)
            test_filename = os.path.join(test_records_path, test_subject_name)

            if (not os.path.isfile(test_filename)):
                file_not_exist_flag = True
            label_filename = os.path.join(test_label_path,
                                          'instance-%s-label.npy' % case_id)
            if (not os.path.isfile(label_filename)):
                file_not_exist_flag = True

        ##################################################################################
        # load test_data
        if file_not_exist_flag:
            print('Loading test_data...')

            # load image and label
            # case name
            # case_name_t1c = case_id + '-t1c.nii.gz'
            # case_name_t1n = case_id + '-t1n.nii.gz'
            # case_name_t2f = case_id + '-t2f.nii.gz'
            # case_name_t2w = case_id + '-t2w.nii.gz'     #BraTS2023
            # brain mask name
            # brain_mask_name = case_id + '-seg.nii.gz'   #BraTS2023

            # case name
            case_name = case_id + '.nii.gz'

            # case_name_t1c = case_id + '_t1ce.nii.gz'
            # case_name_t1n = case_id + '_t1.nii.gz'
            # case_name_t2f = case_id + '_flair.nii.gz'
            # case_name_t2w = case_id + '_t2.nii.gz'
            # brain mask name
            # brain_mask_name = case_id + '_seg.nii.gz'
            brain_mask_name = case_id + '.nii.gz'

            # case path
            case_path = os.path.join(raw_data_dir, case_name)
            # case_path_t1c = os.path.join(raw_data_dir, case_id, case_name_t1c)
            # case_path_t1n = os.path.join(raw_data_dir, case_id, case_name_t1n)
            # case_path_t2f = os.path.join(raw_data_dir, case_id, case_name_t2f)
            # case_path_t2w = os.path.join(raw_data_dir, case_id, case_name_t2w)
            # brain mask path
            # brain_mask_path = os.path.join(raw_data_dir,case_id, brain_mask_name)
            mask_data_dir = raw_data_dir.replace('imagesTr','labelsTr')
            brain_mask_path = os.path.join(mask_data_dir, brain_mask_name)
            # load image and label
            # [_T1c,_T1n,_T2f,_T2w,_label] = load_subject(case_path_t1c,case_path_t1n,case_path_t2f, case_path_t2w,brain_mask_path)
            [_raw, _label] = load_subject(case_path,brain_mask_path)
            # cut edge
            (original_shape, cut_size) = cut_edge(_raw)
            # normalize image
            # _T1c = per_image_normalize(_T1c)
            # _T1n = per_image_normalize(_T1n)
            # _T2f = per_image_normalize(_T2f)
            # _T2w = per_image_normalize(_T2w)
            _raw = per_image_normalize(_raw)

            print('Check original_shape: ', original_shape)
            print('Check cut_size: ', cut_size)
        else:
            continue

        ##################################################################################
        # write records
        if case_id in train_id_list:
            print('Create the training file:')
            write_training_examples(raw=_raw,
                                    label=_label,
                                    original_shape=original_shape,
                                    cut_size=cut_size,
                                    output_file=train_filename)

        if case_id in valid_id_list:
            print('Create the validation file:')
            write_validation_examples(raw=_raw,
                                      original_shape=original_shape,
                                      patch_size=patch_size,
                                      overlap_stepsize=overlap_stepsize,
                                      output_file=valid_filename)
            # write label file
            label_filename = os.path.join(valid_label_path,
                                          'instance-%s-label.npy' % case_id)
            print('Create the converted label file:')  # added
            np.save(label_filename, _label[:, :, :, 0].astype(np.uint8))  # added

        if case_id in test_id_list:
            print('Create the test file:')
            write_test_examples(raw=_raw,
                                original_shape=original_shape,
                                patch_size=patch_size,
                                overlap_stepsize=overlap_stepsize,
                                output_file=test_filename)

            # write label file
            label_filename = os.path.join(test_label_path,
                                          'instance-%s-label.npy' % case_id)
            print('Create the converted label file:')  # added
            np.save(label_filename, _label[:, :, :, 0].astype(np.uint8))  # added

        print('---Done.---')


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



def read_case_ids(file_name,KFold_dir,cv):
    path = os.path.join(KFold_dir,str(cv))
    f = open(os.path.join(path,file_name),"r")
    lines = f.readlines()
    case_id = []
    for i in lines:
        case_id.append(i.replace("\n",""))
    f.close()
    return case_id
def get_train_test_val_id_from_5fold(KFold_dir,cv):
    train_case_id_list =read_case_ids("train.txt",KFold_dir,cv)
    val_case_id_list =read_case_ids("val.txt",KFold_dir,cv)
    test_case_id_list =read_case_ids("test.txt",KFold_dir,cv)
    return train_case_id_list, val_case_id_list, test_case_id_list
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
if __name__ == '__main__':

    ### set parameters
    args = get_args()
    # set random seed
    setup_seed(args.random_seed)
    raw_data_dir = args.raw_data_dir
    KFold_dir = args.KFold_dir
    cv = args.cv
    output_data_dir = args.output_data_dir
    patch_size = args.patch_size
    overlap_step = args.patch_size-args.overlap_step
    # prepare directories
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)

    # pattern for MNI scan case
    # mni_case_pattern = '*_raw_t1_MNI.nii.gz'
    # pattern for refined MNI brain mask
    # mni_brain_mask_pattern = '_brain_mask_MNI.nii.gz'

    train_case_id_list, val_case_id_list, test_case_id_list = get_train_test_val_id_from_5fold(KFold_dir=KFold_dir,cv=cv)

    # pdb.set_trace()
    generate_files(raw_data_dir=raw_data_dir,
                   output_path=output_data_dir,
                   train_id_list=train_case_id_list,
                   valid_id_list=val_case_id_list,
                   test_id_list=test_case_id_list,
                   patch_size=patch_size,
                   overlap_stepsize=overlap_step)

    print('All done.')
