# import cv2
import glob

import numpy as np
from skimage.filters import threshold_otsu, threshold_mean, threshold_local, threshold_isodata
from skimage import morphology
import os
from sklearn.model_selection import KFold, train_test_split
#dataset_name = "fsm_qt1_MNI152_T1_1mm"
def write_file(file_name,k,datalist,idx):
    path = os.path.join("/mnt/SSD2/YHR/dataset/MSD/KFold",str(k))
    os.makedirs(path,exist_ok=True)
    f = open(os.path.join(path,file_name),"w")
    for id in idx:
        f.write(datalist[id]+"\n")
    f.close()




fold_names = []
data_path = "/mnt/HDLV1/YHR/brain_tumor/Task01_BrainTumour/imagesTr"
# folders = glob.glob(os.path.join(data_path,"*_image.nii.gz"))
folders = glob.glob(os.path.join(data_path,'*'))
for fold in folders:
    # fold = os.path.basename(fold).replace("_image.nii.gz","")
    fold = os.path.basename(fold)
    fold_names.append(fold)
print("length of folders is:",len(fold_names))
# print(fold_names)


# print(fold_names)
inds = np.array(range(0,len(fold_names)))
#kf = KFold(n_splits=5,shuffle=False,random_state=1024)
kf = KFold(n_splits=5,shuffle=True)
kf.get_n_splits(inds)
for k, (train_val_index,test_index) in enumerate(kf.split(inds)):
    # print(k)
    train_index, val_index = train_test_split(train_val_index, shuffle=True, train_size=0.99)
    # print(train_index)
    # print(val_index)
    # print(test_index)
    write_file("train.txt", k, fold_names, train_index)
    write_file("val.txt", k, fold_names, val_index)
    write_file("test.txt", k, fold_names, test_index)


    
