import glob

import torch
import os
import numpy as np
import h5py
from torch.utils.data import Dataset
import pdb

class BrainDatasetVal(Dataset):
    """
    Brain dataset for prediction
    The dataset contains only one test instance
    """
    def __init__(self, root_dir,
                 test_instance_id, patch_size,
                 overlap_step, transform=None):
        """
        :param root_dir (string): Directory with all the scan folders
        :param transform (callable, optional): Optional transform
               to be applied on a sample.
               test_instance_id: scalar, id of test instance
               patch_size: size of patch
               overlap_step: step of overlap
        """
        self.root_dir = root_dir
        self.overlap_step = overlap_step
        self.patch_size = patch_size
        # self.scans = ['%s-test-overlap-%d-patch-%d.hdf5'
        #               % (test_instance_id, overlap_step, patch_size)]
        self.scans = glob.glob(os.path.join(self.root_dir,'*-valid-overlap-%d-patch-%d.hdf5'% (overlap_step, patch_size)))
        self.label_dir = os.path.join(self.root_dir,"../","val_label")
        self.pattern = '-valid-overlap-%d-patch-%d.hdf5'% (overlap_step, patch_size)
        self.transform = transform

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        scan_name = os.path.join(self.root_dir,
                                 self.scans[idx])
        # open file
        f = h5py.File(scan_name, 'r')
        # get HDF5 dataset
        # image = f['image']
        image_input_t1c = f['image_t1c']
        image_input_t1n = f['image_t1n']
        image_input_t2f = f['image_t2f']
        image_input_t2w = f['image_t2w']


        patch_id_t1c = f['patch_id_t1c']
        original_shape = f['original_shape']

        # get test_data from dataset
        # image = image[:]
        image_input_t1c = image_input_t1c[:]
        image_input_t1n = image_input_t1n[:]
        image_input_t2f = image_input_t2f[:]
        image_input_t2w = image_input_t2w[:]


        original_shape = original_shape[:]
        patch_id_t1c = patch_id_t1c[:]


        file_id= os.path.basename(scan_name).replace(self.pattern,"")
        # print('Loading label...')
        label_file = os.path.join(self.label_dir, '%s-label.npy' % file_id)
        label = np.load(label_file)

        # # BraTS2023之前的需要把4值改为3值
        # label[label == 4] = 3
        # label = label.astype(np.int64)
        # print('Check label: ', label.shape, np.max(label))
        # label = torch.unsqueeze(label,dim=2)

        # sample is a dictionary
        # image is the container of chuncks
        sample = {'image_t1c': image_input_t1c, 'image_t1n': image_input_t1n,'image_t2f': image_input_t2f,'image_t2w': image_input_t2w,'label': label,'patch_id': patch_id_t1c,'original_shape': original_shape}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Normalize(object):
    """Normalize per image
       https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
    """
    def __call__(self, sample):
        image_t1c = sample['image_t1c']
        image_t1n = sample['image_t1n']
        image_t2f = sample['image_t2f']
        image_t2w = sample['image_t2w']
        patch_id_t1c = sample['patch_id_t1c']
        original_shape = sample['original_shape']
        # get mean
        mean_t1c = np.mean(image_t1c)
        mean_t1n = np.mean(image_t1n)
        mean_t2f = np.mean(image_t2f)
        mean_t2w = np.mean(image_t2w)
        # get adjusted_stddev
        stddev_t1c = np.std(image_t1c)
        stddev_t1n = np.std(image_t1n)
        stddev_t2f = np.std(image_t2f)
        stddev_t2w = np.std(image_t2w)
        voxel_dev = 1.0 / np.sqrt(image_t1c.shape[0] *
                                  image_t1c.shape[1] *
                                  image_t1c.shape[2])
        adjusted_stddev_t1c = max(stddev_t1c, voxel_dev)
        adjusted_stddev_t1n = max(stddev_t1n, voxel_dev)
        adjusted_stddev_t2f = max(stddev_t2f, voxel_dev)
        adjusted_stddev_t2w = max(stddev_t2w, voxel_dev)
        # normalize
        image_t1c = (image_t1c - mean_t1c) / adjusted_stddev_t1c
        image_t1n = (image_t1n - mean_t1n) / adjusted_stddev_t1n
        image_t2f = (image_t2f - mean_t2f) / adjusted_stddev_t2f
        image_t2w = (image_t2w - mean_t2w) / adjusted_stddev_t2w

        # convert test_data type of image to float32
        image_t1c = image_t1c.astype(np.float32)
        image_t1n = image_t1n.astype(np.float32)
        image_t2f = image_t2f.astype(np.float32)
        image_t2w = image_t2w.astype(np.float32)

        return {'image_t1c': image_t1c, 'image_t1n': image_t1n,'image_t2f': image_t2f,'image_t2w': image_t2w,'patch_id_t1c': patch_id_t1c,'original_shape': original_shape}

class ValToTensor(object):
    """Convert ndarrys in sample to Tensors."""

    def __call__(self, sample):
        image_t1c = sample['image_t1c']
        image_t1n = sample['image_t1n']
        image_t2f = sample['image_t2f']
        image_t2w = sample['image_t2w']

        patch_id = sample['patch_id']
        original_shape = sample['original_shape']

        label = sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image_t1c = image_t1c.transpose((3, 0, 1, 2))
        image_t1n = image_t1n.transpose((3, 0, 1, 2))
        image_t2f = image_t2f.transpose((3, 0, 1, 2))
        image_t2w = image_t2w.transpose((3, 0, 1, 2))
        # label = label.transpose((3, 0, 1, 2))

        return {'image_t1c': torch.from_numpy(image_t1c),
                 'image_t1n': torch.from_numpy(image_t1n),
                 'image_t2f': torch.from_numpy(image_t2f),
                 'image_t2w': torch.from_numpy(image_t2w),
                'label':torch.unsqueeze(torch.from_numpy(label),dim=0),
                'patch_id': torch.from_numpy(patch_id),
                'original_shape': torch.from_numpy(original_shape)}