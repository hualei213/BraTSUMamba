import h5py
import numpy as np
import os
import nibabel as nib
import sys
import pdb
sys.path.append('../')
from config import get_args

""" Generate hdf5 records
"""

def load_subject(raw_data_dir, subject_id):
    '''Load subject test_data.

    Args:
        subject_id: [1-23]

    Returns:
        [T1, T2, label]
        [T1, label]
    '''

    # subject_name = 'subject-%d-' % subject_id
    subject_name = '%d' % subject_id

    # load brain test_data
    # fl = os.path.join(raw_data_dir, subject_name, 'I.nii.gz')
    fl = os.path.join(raw_data_dir, subject_name, 'I_standard.nii.gz')
    img_T1 = nib.load(fl)
    inputs_T1 = img_T1.get_data()

    # expand inputs_T1 as 4 dimensional, which is required by conv3D @ tensorflow
    inputs_T1 = np.expand_dims(inputs_T1, axis=3)

    # file path for mask of brain
    # fl_brain = os.path.join(raw_data_dir, subject_name, 'I_brain.nii.gz')
    fl_brain = os.path.join(raw_data_dir, subject_name, 'brainmask_standard_refined.nii.gz')
    # brain mask
    img_brain_mask = nib.load(fl_brain)
    label_brain_mask = img_brain_mask.get_data()
    # cast to uint
    label_brain_mask = label_brain_mask.astype(np.uint8)

    # expand inputs_label as 4 dimensional, which is required by conv3D @ tensorflow
    inputs_label = np.expand_dims(label_brain_mask, axis=3)

    return [inputs_T1, inputs_label]

def prepare_validation(cutted_image, patch_size, overlap_stepsize):
    """Determine patches for validation."""

    patch_ids = []

    D, H, W, _ = cutted_image.shape

    drange = list(range(0, D - patch_size + 1, overlap_stepsize))
    hrange = list(range(0, H - patch_size + 1, overlap_stepsize))
    wrange = list(range(0, W - patch_size + 1, overlap_stepsize))

    if (D - patch_size) % overlap_stepsize != 0:
        drange.append(D - patch_size)
    if (H - patch_size) % overlap_stepsize != 0:
        hrange.append(H - patch_size)
    if (W - patch_size) % overlap_stepsize != 0:
        wrange.append(W - patch_size)

    for d in drange:
        for h in hrange:
            for w in wrange:
                patch_ids.append((d, h, w))

    return patch_ids

def write_training_examples(T1c, T1n, T2f, T2w,label, original_shape, cut_size, output_file):
    """Create a training hdf5 file.

    Args:
        T1: T1 image. [Depth, Height, Width, 1].
        label: Label. [Depth, Height, Width, 1].
        original_shape: A list of three integers [D, H, W].
        cut_size: A list of six integers
            [Depth_s, Depth_e, Height_s, Height_e, Width_s, Width_e].
        output_file: The file name for the hdf5 file.
    """
    f = h5py.File(output_file, "w")
    dset_image_t1c = f.create_dataset(name="image_t1c",
                                  data=T1c)
    dset_image_t1n = f.create_dataset(name="image_t1n",
                                      data=T1n)
    dset_image_t2f = f.create_dataset(name="image_t2f",
                                      data=T2f)
    dset_image_t2w = f.create_dataset(name="image_t2w",
                                      data=T2w)

    dset_label = f.create_dataset(name="label",
                                  data=label)
    dset_original_shape = f.create_dataset(name="original_shape",
                                           data=original_shape)
    dset_cut_size = f.create_dataset(name="cut_size",
                                     data=cut_size)

    f.close()

    return


def write_validation_examples(T1c, T1n, T2f, T2w,
                        original_shape,
                        patch_size,
                        overlap_stepsize,
                        output_file):
    """Create a testing hdf5 file.

    Args:
        T1: T1 image. [Depth, Height, Width, 1].
        patch_size: An integer.
        overlap_stepsize: An integer.
        output_file: The file name for the hdf5 file.
    """
    # T1 = T1[cut_size[0]:cut_size[1], cut_size[2]:cut_size[3], cut_size[4]:cut_size[5], :]
    patch_ids_t1c = prepare_validation(T1c, patch_size, overlap_stepsize)
    patch_ids_t1n = prepare_validation(T1n, patch_size, overlap_stepsize)
    patch_ids_t2f = prepare_validation(T2f, patch_size, overlap_stepsize)
    patch_ids_t2w = prepare_validation(T2w, patch_size, overlap_stepsize)



    print('Number of patches:', len(patch_ids_t1c))

    f = h5py.File(output_file, "w")
    dset_patch_size = f.create_dataset(name="patch_size",
                                       data=patch_size)
    dset_overlap_stepsize = f.create_dataset(name="overlap_stepsize",
                                             data=overlap_stepsize)
    dset_patch_ids_t1c = f.create_dataset(name="patch_id_t1c",
                                      data=patch_ids_t1c)



    dset_original_shape = f.create_dataset(name="original_shape",
                                           data=original_shape)
    # chuncked storage
    dset_image_t1c = f.create_dataset(name="image_t1c",
                                  shape=(patch_size,
                                         patch_size,
                                         patch_size * len(patch_ids_t1c),
                                         1),
                                  chunks=(patch_size,
                                          patch_size,
                                          patch_size,
                                          1),
                                  dtype=T1c.dtype)
    dset_image_t1n = f.create_dataset(name="image_t1n",
                                      shape=(patch_size,
                                             patch_size,
                                             patch_size * len(patch_ids_t1n),
                                             1),
                                      chunks=(patch_size,
                                              patch_size,
                                              patch_size,
                                              1),
                                      dtype=T1n.dtype)
    dset_image_t2f = f.create_dataset(name="image_t2f",
                                      shape=(patch_size,
                                             patch_size,
                                             patch_size * len(patch_ids_t2f),
                                             1),
                                      chunks=(patch_size,
                                              patch_size,
                                              patch_size,
                                              1),
                                      dtype=T2f.dtype)
    dset_image_t2w = f.create_dataset(name="image_t2w",
                                      shape=(patch_size,
                                             patch_size,
                                             patch_size * len(patch_ids_t2w),
                                             1),
                                      chunks=(patch_size,
                                              patch_size,
                                              patch_size,
                                              1),
                                      dtype=T2w.dtype)
        # write chunck by chunck
    for i in range(len(patch_ids_t1c)):
        (d, h, w) = patch_ids_t1c[i]
        _T1c = T1c[d:d + patch_size, h:h + patch_size, w:w + patch_size, :]
        dset_image_t1c[0:patch_size, 0:patch_size, i*patch_size:(i+1)*patch_size, :] = _T1c
    for i in range(len(patch_ids_t1n)):
        (d, h, w) = patch_ids_t1n[i]
        _T1n = T1n[d:d + patch_size, h:h + patch_size, w:w + patch_size, :]
        dset_image_t1n[0:patch_size, 0:patch_size, i * patch_size:(i + 1) * patch_size, :] = _T1n
    for i in range(len(patch_ids_t2f)):
        (d, h, w) = patch_ids_t2f[i]
        _T2f = T2f[d:d + patch_size, h:h + patch_size, w:w + patch_size, :]
        dset_image_t2f[0:patch_size, 0:patch_size, i * patch_size:(i + 1) * patch_size, :] = _T2f
    for i in range(len(patch_ids_t2w)):
        (d, h, w) = patch_ids_t2w[i]
        _T2w = T2w[d:d + patch_size, h:h + patch_size, w:w + patch_size, :]
        dset_image_t2w[0:patch_size, 0:patch_size, i * patch_size:(i + 1) * patch_size, :] = _T2w
    f.close()



def write_test_examples(T1c, T1n, T2f, T2w,
                        original_shape,
                        patch_size,
                        overlap_stepsize,
                        output_file):
    """Create a testing hdf5 file.

    Args:
        T1: T1 image. [Depth, Height, Width, 1].
        patch_size: An integer.
        overlap_stepsize: An integer.
        output_file: The file name for the hdf5 file.
    """
    # T1 = T1[cut_size[0]:cut_size[1], cut_size[2]:cut_size[3], cut_size[4]:cut_size[5], :]

    patch_ids_t1c = prepare_validation(T1c, patch_size, overlap_stepsize)
    patch_ids_t1n = prepare_validation(T1n, patch_size, overlap_stepsize)
    patch_ids_t2f = prepare_validation(T2f, patch_size, overlap_stepsize)
    patch_ids_t2w = prepare_validation(T2w, patch_size, overlap_stepsize)

    print('Number of patches:', len(patch_ids_t1c))

    f = h5py.File(output_file, "w")
    dset_patch_size = f.create_dataset(name="patch_size",
                                       data=patch_size)
    dset_overlap_stepsize = f.create_dataset(name="overlap_stepsize",
                                             data=overlap_stepsize)
    # dset_patch_ids = f.create_dataset(name="patch_id",
    #                                   data=patch_ids)

    dset_patch_ids_t1c = f.create_dataset(name="patch_id_t1c",
                                      data=patch_ids_t1c)




    dset_original_shape = f.create_dataset(name="original_shape",
                                           data=original_shape)
    # chuncked storage
    dset_image_t1c = f.create_dataset(name="image_t1c",
                                      shape=(patch_size,
                                             patch_size,
                                             patch_size * len(patch_ids_t1c),
                                             1),
                                      chunks=(patch_size,
                                              patch_size,
                                              patch_size,
                                              1),
                                      dtype=T1c.dtype)
    dset_image_t1n = f.create_dataset(name="image_t1n",
                                      shape=(patch_size,
                                             patch_size,
                                             patch_size * len(patch_ids_t1n),
                                             1),
                                      chunks=(patch_size,
                                              patch_size,
                                              patch_size,
                                              1),
                                      dtype=T1n.dtype)
    dset_image_t2f = f.create_dataset(name="image_t2f",
                                      shape=(patch_size,
                                             patch_size,
                                             patch_size * len(patch_ids_t2f),
                                             1),
                                      chunks=(patch_size,
                                              patch_size,
                                              patch_size,
                                              1),
                                      dtype=T2f.dtype)
    dset_image_t2w = f.create_dataset(name="image_t2w",
                                      shape=(patch_size,
                                             patch_size,
                                             patch_size * len(patch_ids_t2w),
                                             1),
                                      chunks=(patch_size,
                                              patch_size,
                                              patch_size,
                                              1),
                                      dtype=T2w.dtype)
    # write chunck by chunck
    for i in range(len(patch_ids_t1c)):
        (d, h, w) = patch_ids_t1c[i]
        _T1c = T1c[d:d + patch_size, h:h + patch_size, w:w + patch_size, :]
        dset_image_t1c[0:patch_size, 0:patch_size, i * patch_size:(i + 1) * patch_size, :] = _T1c
    for i in range(len(patch_ids_t1n)):
        (d, h, w) = patch_ids_t1n[i]
        _T1n = T1n[d:d + patch_size, h:h + patch_size, w:w + patch_size, :]
        dset_image_t1n[0:patch_size, 0:patch_size, i * patch_size:(i + 1) * patch_size, :] = _T1n
    for i in range(len(patch_ids_t2f)):
        (d, h, w) = patch_ids_t2f[i]
        _T2f = T2f[d:d + patch_size, h:h + patch_size, w:w + patch_size, :]
        dset_image_t2f[0:patch_size, 0:patch_size, i * patch_size:(i + 1) * patch_size, :] = _T2f
    for i in range(len(patch_ids_t2w)):
        (d, h, w) = patch_ids_t2w[i]
        _T2w = T2w[d:d + patch_size, h:h + patch_size, w:w + patch_size, :]
        dset_image_t2w[0:patch_size, 0:patch_size, i * patch_size:(i + 1) * patch_size, :] = _T2w
    f.close()

def cut_edge(data):
    '''Cuts zero edge for a 3D image.

    Args:
        data: A 3D image, [Depth, Height, Width, 1].

    Returns:
        original_shape: [Depth, Height, Width]
        cut_size: A list of six integers [Depth_s, Depth_e, Height_s, Height_e, Width_s, Width_e]
    '''

    D, H, W, _ = data.shape
    D_s, D_e = 0, D - 1
    H_s, H_e = 0, H - 1
    W_s, W_e = 0, W - 1

    while D_s < D:
        if data[D_s].sum() != 0:
            break
        D_s += 1
    while D_e > D_s:
        if data[D_e].sum() != 0:
            break
        D_e -= 1
    while H_s < H:
        if data[:, H_s].sum() != 0:
            break
        H_s += 1
    while H_e > H_s:
        if data[:, H_e].sum() != 0:
            break
        H_e -= 1
    while W_s < W:
        if data[:, :, W_s].sum() != 0:
            break
        W_s += 1
    while W_e > W_s:
        if data[:, :, W_e].sum() != 0:
            break
        W_e -= 1

    original_shape = [D, H, W]
    cut_size = [int(D_s), int(D_e + 1), int(H_s), int(H_e + 1), int(W_s), int(W_e + 1)]
    return (original_shape, cut_size)

def per_image_normalize(input_image):
    """
    Normalize per image
       https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization

    :param input_image: 3D image.
    :return: Normalized image.
    """

    # get mean
    mean = np.mean(input_image)
    # get adjusted_stddev
    stddev = np.std(input_image)
    voxel_dev = 1.0 / np.sqrt(input_image.shape[0] *
                              input_image.shape[1] *
                              input_image.shape[2])
    adjusted_stddev = max(stddev, voxel_dev)
    # normalize
    output_image = (input_image - mean) / adjusted_stddev

    # convert test_data type of image to float32
    output_image = output_image.astype(np.float32)

    return output_image

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


    all_id_list = train_id_list + valid_id_list + test_id_list    # get id list of all subjects
    for i in all_id_list:
        print('---Process subject %d:---' % i)

        file_not_exist_flag = False
        if i in train_id_list:
            train_subject_name = 'instance-%d.hdf5' % i
            train_filename = os.path.join(train_records_path, train_subject_name)
            if (not os.path.isfile(train_filename)):
                file_not_exist_flag = True

        if i in valid_id_list:
            valid_subject_name = 'instance-%d-valid-overlap-%d-patch-%d.hdf5' \
                                 % (i, overlap_stepsize, patch_size)
            valid_filename = os.path.join(valid_records_path, valid_subject_name)
            if (not os.path.isfile(valid_filename)):
                file_not_exist_flag = True
        if i in test_id_list:
            test_subject_name = 'instance-%d-test-overlap-%d-patch-%d.hdf5' \
                                % (i, overlap_stepsize, patch_size)
            test_filename = os.path.join(test_records_path, test_subject_name)

            if (not os.path.isfile(test_filename)):
                file_not_exist_flag = True
            label_filename = os.path.join(test_label_path,
                                          'instance-%d-label.npy' % i)
            if (not os.path.isfile(label_filename)):
                file_not_exist_flag = True

        ##################################################################################
        # load test_data
        if file_not_exist_flag:
            print('Loading test_data...')
            # load image and label
            [_T1, _label] = load_subject(raw_data_dir, i)
            # cut edge
            (original_shape, cut_size) = cut_edge(_T1)
            # normalize image
            _T1 = per_image_normalize(_T1)

            print('Check original_shape: ', original_shape)
            print('Check cut_size: ', cut_size)
        else:
            continue

        ##################################################################################
        # write records
        if i in train_id_list:
            print('Create the training file:')
            write_training_examples(T1 = _T1,
                                    label = _label,
                                    original_shape = original_shape,
                                    cut_size = cut_size,
                                    output_file = train_filename)

        if i in valid_id_list:
            print('Create the validation file:')
            write_validation_examples(T1 = _T1,
                                      label = _label,
                                      patch_size = patch_size,
                                      original_shape = original_shape,
                                      cut_size = cut_size,
                                      overlap_stepsize = overlap_stepsize,
                                      output_file = valid_filename)

            # label_filename = os.path.join(valid_records_path,
            #                                         'instance-%d-label.npy' % i)
            # print('Create the converted label file:')
            # np.save(label_filename, _label[:, :, :, 0].astype(np.uint8))

        if i in test_id_list:
            print('Create the test file:')
            write_test_examples(T1 = _T1,
                                original_shape = original_shape,
                                patch_size = patch_size,
                                cut_size = cut_size,
                                overlap_stepsize = overlap_stepsize,
                                output_file = test_filename)

            # write label file
            label_filename = os.path.join(test_label_path,
                                          'instance-%d-label.npy' % i)
            print('Create the converted label file:')  # added
            np.save(label_filename, _label[:, :, :, 0].astype(np.uint8))  # added

        print('---Done.---')


if __name__ == '__main__':

    # set parameters
    args = get_args()
    raw_data_dir = args.raw_data_dir
    output_data_dir = args.output_data_dir
    patch_size = args.patch_size
    overlap_step = args.overlap_step
