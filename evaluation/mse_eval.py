import os

import numpy as np

def getImage(file_path):
    original_list=[]
    register_list=[]
    image_lists = sorted(os.listdir(file_path))
    for file_name in image_lists:
        if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
            original_list.append(file_name)
            register_list.append(file_name)


    return original_list,register_list


def calculate_mse(image_a, image_b):
    # Ensure the images have the same shape
    assert image_a.shape == image_b.shape, "Images must have the same dimensions"

    # Calculate squared differences
    squared_diff = np.square(image_a - image_b)

    # Calculate mean squared error
    mse = np.mean(squared_diff)

    return mse


if __name__ == '__main__':
    original_path = ""
    register_path = ""

