import os

import nibabel as nib
import numpy as np


def convert(original_nifty_file,
            numpy_file_name,
            destination_file_name):

    # load original nifty file
    img_org = nib.load(original_nifty_file)
    # get numpy array
    np_arr = np.load(numpy_file_name)
    # create nifty image
    new_img = nib.Nifti1Image(dataobj=np_arr,
                              affine=img_org.affine,
                              header=img_org.header)
    # save image
    # nib.save(img=new_img,
    #          filename=destination_file_name)


if __name__=="__main__":
    test_id = "35_35"
    checkpoint=990
    raw_nifty_file = os.path.join("/media/sggq/MyDisk/datasets/HTU_BrainMask/brain_mask/brain_raw_mask_MNI/",
                                  test_id + '_raw_t1_MNI.nii.gz')
    np_arr_file = os.path.join("/result/HTU_Brainmask/result-ModifyUnet_HTU_Brainmask_1000",
                               'test_instance-%s_checkpoint_%d.npy' \
                               % (test_id, checkpoint))
    dst_file_name = os.path.join("./", '%s_predict_I_brain.nii.gz' % test_id)
    convert(raw_nifty_file, np_arr_file, dst_file_name)