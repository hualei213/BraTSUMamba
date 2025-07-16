import os

import nibabel as nib
import numpy as np


def get_all_subdirectories(directory):
    # 获取所有子目录名
    subdirectories = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    return subdirectories


def transform(file_path):
    # 加载 NIfTI 文件
    img = nib.load(file_path)
    data = img.get_fdata()

    # 打印初始数据值范围（可选）
    print("Initial unique values in the data:", np.unique(data))

    # 将值为4的标签替换为3
    data[data == 4] = 3
    data = data.astype(np.int64)

    # 打印修改后的数据值范围（可选）
    print("Modified unique values in the data:", np.unique(data))

    # 创建新的 NIfTI 图像
    new_img = nib.Nifti1Image(data, img.affine, img.header)

    # 保存修改后的 NIfTI 文件
    nib.save(new_img, file_path)

    print(f"Modified NIfTI file saved to {file_path}")


if __name__ == '__main__':

    directory_path = '/mnt/SSD2/YHR/dataset/MICCAI_BraTS2020_TrainingData_seg4to3'
    subdirs = get_all_subdirectories(directory_path)
    for subdir in subdirs:
        file_path = os.path.join(directory_path,subdir,subdir+'_seg.nii.gz')
        transform(file_path)
    print("All complete")
    # print(subdirs)

