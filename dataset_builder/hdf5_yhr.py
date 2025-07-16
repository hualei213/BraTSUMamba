import os

import nibabel as nib
import numpy as np
import sys



sys.path.append('../')

from hdf5_generator import write_training_examples, \
    write_validation_examples, \
    write_test_examples, \
    cut_edge, \
    per_image_normalize



def load_subject(case_path_t1c,case_path_t1n,case_path_t2f,case_path_t2w):
    """
    load case related test_data
    :param case_path: path to case
    :param brain_mask_path: path to brain mask
    :return: [T1, label]
    """

    # load brain test_data
    img_T1c = nib.load(case_path_t1c)
    img_T1n = nib.load(case_path_t1n)
    img_T2f = nib.load(case_path_t2f)
    img_T2w = nib.load(case_path_t2w)

    inputs_T1c = img_T1c.get_fdata()
    inputs_T1n = img_T1n.get_fdata()
    inputs_T2f = img_T2f.get_fdata()
    inputs_T2w = img_T2w.get_fdata()

    # expand inputs_T1 as 4 dimensional, which is required by conv3D @ tensorflow
    inputs_T1c = np.expand_dims(inputs_T1c, axis=3)
    inputs_T1n = np.expand_dims(inputs_T1n, axis=3)
    inputs_T2f = np.expand_dims(inputs_T2f, axis=3)
    inputs_T2w = np.expand_dims(inputs_T2w, axis=3)

    # # get case id
    # img_label = nib.load(brain_mask_path)
    # inputs_label = img_label.get_fdata()
    #
    # # expand dimensions
    # inputs_label = np.expand_dims(inputs_label, axis=3)
    # # cast to uint8
    # inputs_label = inputs_label.astype(np.uint8)

    # pdb.set_trace()

    return inputs_T1c,inputs_T1n,inputs_T2f,inputs_T2w


# 指定要读取的txt文件路径
file_path = '/mnt/HDD1/YHR/BraTS2023-GLI-Traning.txt'

train_records_path= "/mnt/HDD1/YHR/BraTS2023-GLI-Training-flirt-hdf5"


# 打开文件
with open(file_path, 'r') as file:
    # 逐行读取文件
    for line in file:
        # 打印每一行的内容
        line=line.strip()  # 去除行尾的换行符
        file_dir = line




        file_name = os.path.basename(file_dir)
        case_id = file_name
        # case_id = file_name.replace("_image.nii.gz","")

        # case name
        case_name_t1c = case_id + '-t1c.nii.gz'
        case_name_t1n = case_id + '-t1n.nii.gz'
        case_name_t2f = case_id + '-t2f.nii.gz'
        case_name_t2w = case_id + '-t2w.nii.gz'

        # case path
        # case_path = os.path.join(raw_data_dir, case_name)
        case_path_t1c = os.path.join(file_dir, case_name_t1c)
        case_path_t1n = os.path.join(file_dir, case_name_t1n)
        case_path_t2f = os.path.join(file_dir, case_name_t2f)
        case_path_t2w = os.path.join(file_dir, case_name_t2w)



        train_subject_name = 'instance-%s.hdf5' % case_id
        train_filename = os.path.join(train_records_path, train_subject_name)


        # case_path = os.path.join(raw_data_dir, case_name)
        # case_path = file_dir
        # brain mask path
        # brain_mask_path = os.path.join(raw_data_dir, brain_mask_name)
        # brain_mask_path = case_path.replace("img_nii.gz","mask_nii.gz")
        # load image and label
        # [_T1, _label] = load_subject(case_path, brain_mask_path)
        [_T1c, _T1n, _T2f, _T2w] = load_subject(case_path_t1c, case_path_t1n, case_path_t2f, case_path_t2w)
        # cut edge
        (original_shape, cut_size) = cut_edge(_T1c)
        # normalize image
        _T1c = per_image_normalize(_T1c)
        _T1n = per_image_normalize(_T1n)
        _T2f = per_image_normalize(_T2f)
        _T2w = per_image_normalize(_T2w)

        write_training_examples(T1c=_T1c, T1n=_T1n, T2f=_T2f, T2w=_T2w,
                                original_shape=original_shape,
                                cut_size=cut_size,
                                output_file=train_filename)

        # write_training_examples(T1=_T1, label=_label, original_shape=original_shape, cut_size=cut_size,
        #                         output_file=train_filename)

        print(f"{case_id} have Done")




