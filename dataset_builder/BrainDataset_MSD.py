import torch
import os
import numpy as np
import h5py
import time
from torch.utils.data import Dataset
import pdb
import nibabel as nib


class BrainDataset_MSD(Dataset):
    """
    Brain dataset
    """

    def __init__(self, root_dir, transform=None):
        # flirt_dir,

        """
        :param root_dir (string): Directory with all the scan folders
        :param transform (callable, optional): Optional transform
               to be applied on a sample.
        """
        self.root_dir = root_dir
        self.scans_input = os.listdir(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.scans_input)

    def __getitem__(self, idx):
        # begin_time = time.time()

        scan_name_input = os.path.join(self.root_dir, self.scans_input[idx])
        # scan_name_flirt = os.path.join(self.flirt_dir,self.scans_input[idx])
        # scan_name_atlas = os.path.join(self.atlas_dir,self.atlas_input[0])

        # file_name = self.scans_input[idx][self.scans_input[idx].find("instance-") + len("instance-"):self.scans_input[idx](".hdf5")]
        # scan_name_flirt = os.path.join(self.flirt_dir,file_name, self.scans_input[idx])
        # scan_name_atlas = os.path.join(self.atlas_dir,self.scans_atlas[0])

        # open input file
        f = h5py.File(scan_name_input, 'r')
        # get HDF5 dataset
        image_input = f['image']

        label_input = f['label']



        image_input = image_input[:]
        label_input = label_input[:]

        #BraTS2023之前的需要把4值改为3值
        # label_input[label_input == 4] = 3
        # label_input = label_input.astype(np.int64)





        sample_input = {'image': image_input, 'label': label_input}

        # Resize and Totensor
        if self.transform:
            sample_input = self.transform(sample_input)

        sample_input["id"] = self.scans_input[idx]
        return sample_input


class Normalize_MSD(object):
    """Normalize per image
       https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
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

        return {'image': image, 'label': label}


class RandomCrop_MSD(object):
    """
    Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired outupt size. If int,
        square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

    def __call__(self, sample_input):
        image = sample_input['image']

        label = sample_input['label']



        h, w, d = image.shape[:3]

        new_h, new_w, new_d = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        back = np.random.randint(0, d - new_d)

        image = image[top: top + new_h,
                left: left + new_w,
                back: back + new_d]

        label = label[top: top + new_h,
                left: left + new_w,
                back: back + new_d]

        return {'image': image, 'label': label}



class HorizenFlip_MSD(object):
    """
    horizentally flip the image, with given probability p.
    0 < p < 1
    Depth / channel of the image is unchanged.
    """

    def __init__(self, p):
        assert 0 <= p and p <= 1
        self.prob = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # generate random number
        rand_prob = np.random.uniform(0.0, 1.0)

        if rand_prob <= self.prob:
            # flip. Refer https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html
            image = np.flip(image, axis=2).copy()
            label = np.flip(label, axis=2).copy()

        return {'image': image, 'label': label}


class ToTensor_MSD(object):
    """Convert ndarrys in sample to Tensors."""

    def __call__(self, sample_input):
        image = sample_input['image']
        label = sample_input['label']


        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((3, 0, 1, 2))
        label = label.transpose((3, 0, 1, 2))





        return ({'image': torch.from_numpy(image),
                 'label': torch.from_numpy(label)
                 }
        )

