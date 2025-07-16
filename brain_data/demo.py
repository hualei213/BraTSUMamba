import glob
import nibabel as nib
import numpy as np
"""
    维度转换 
"""
def trans(image_path):
    img = nib.load(image_path)
    img_affine = img.affine
    img = img.get_data()
    img = img.transpose(0, 2, 1, 3)
    img_affine[:, [1, 2]] = img_affine[:, [2, 1]]
    nib.Nifti1Image(img, img_affine).to_filename(image_path)


if __name__ == '__main__':
    image_path = './data_esop/image/IBSR_01_ana_brainmask.nii.gz'
    file_name = glob.glob("/media/sggq/MyDisk/datasets/IBSR/raw_data/brain2 (copy)/*_brain.nii.gz")
    for file in file_name:
        trans(file)
