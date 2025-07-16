import torch
import os
import numpy as np
import h5py
from torch.utils.data import Dataset
import pdb

class BrainDatasetTest_BraTS2020(Dataset):
    """
    Brain dataset for prediction
    The dataset contains only one test instance
    """
    def __init__(self, root_dir,test_instance_id, patch_size,
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
        # self.atlas_dir = atlas_dir
        # self.atlas_input = os.listdir(atlas_dir)
        self.scans = ['%s-test-overlap-%d-patch-%d.hdf5'
                      % (test_instance_id, overlap_step, patch_size)]
        self.transform = transform

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        scan_name = os.path.join(self.root_dir,
                                 self.scans[idx])
        # scan_name_atlas = os.path.join(self.atlas_dir, self.atlas_input[0])

        # open file
        f = h5py.File(scan_name, 'r')
        # get HDF5 dataset
        image_t1c = f['image_t1c']
        image_t1n = f['image_t1n']
        image_t2f = f['image_t2f']
        image_t2w = f['image_t2w']

        patch_id_t1c = f['patch_id_t1c']

        original_shape = f['original_shape']


        #get test_data from dataset
        image_t1c = image_t1c[:]
        image_t1n = image_t1n[:]
        image_t2f = image_t2f[:]
        image_t2w = image_t2w[:]
        patch_id = patch_id_t1c[:]
        original_shape = original_shape[:]

        # sample is a dictionary
        # image is the container of chuncks
        sample = {'image_t1c': image_t1c,
                  'image_t1n': image_t1n,
                  'image_t2f': image_t2f,
                  'image_t2w': image_t2w,
                  'patch_id': patch_id,
                  'original_shape': original_shape}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Normalize_BraTS2020(object):
    """Normalize per image
       https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
    """
    def __call__(self, sample):
        image, patch_id = sample['image'], sample['patch_id']
        original_shape = sample['original_shape']
        # get mean
        mean = np.mean(image)
        # get adjusted_stddev
        stddev = np.std(image)
        voxel_dev = 1.0 / np.sqrt(image.shape[0] *
                                  image.shape[1] *
                                  image.shape[2])
        adjusted_stddev = max(stddev, voxel_dev)
        # normalize
        image = (image - mean) / adjusted_stddev

        # convert test_data type of image to float32
        image = image.astype(np.float32)

        return {'image': image,
                'patch_id': patch_id,
                'original_shape': original_shape}

class TestToTensor_BraTS2020(object):
    """Convert ndarrys in sample to Tensors."""

    def __call__(self, sample):
        image_t1c = sample['image_t1c']
        image_t1n = sample['image_t1n']
        image_t2f = sample['image_t2f']
        image_t2w = sample['image_t2w']




        patch_id = sample['patch_id']
        original_shape = sample['original_shape']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image_t1c = image_t1c.transpose((3, 0, 1, 2))
        image_t1n = image_t1n.transpose((3, 0, 1, 2))
        image_t2f = image_t2f.transpose((3, 0, 1, 2))
        image_t2w = image_t2w.transpose((3, 0, 1, 2))



        return {'image_t1c': torch.from_numpy(image_t1c),
                 'image_t1n': torch.from_numpy(image_t1n),
                 'image_t2f': torch.from_numpy(image_t2f),
                 'image_t2w': torch.from_numpy(image_t2w),
                'patch_id': torch.from_numpy(patch_id),
                'original_shape': torch.from_numpy(original_shape)
                }