import glob
import os.path

import nibabel as nib
import numpy as np


def data_transform(file_path):
    # load file
    old_img = nib.load(file_path)
    # get file test_data
    old_data = old_img.get_fdata()
    # test_data number > 0 set 1
    new_data = np.int64(old_data > 0)
    # create new image
    new_img = nib.Nifti1Image(dataobj=new_data,
                              affine=old_img.affine,
                              header=old_img.header)
    # save image
    nib.save(img=new_img,
             filename=file_path)
    print("done")


if __name__ == "__main__":
    # file_dir="/media/sggq/MyDisk/实验结果/IBSR/张道强/ALL/"
    file_dir="/media/sggq/MyDisk/datasets/IBSR/brain_mask/brain_mask_raw/"
    file_list = glob.glob(os.path.join(file_dir,"*/*_mask.nii.gz"))
    count = 1
    print("total files :{}".format(len(file_list)))
    for file_path in file_list:
        print("process:{}/{}  file_name-->{}".format(count, len(file_list), file_path))
        count += 1
        # process test_data transform
        data_transform(file_path)
    print("All done")

