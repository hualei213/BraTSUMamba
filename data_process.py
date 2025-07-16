import glob
import shutil
import os.path

def file_copy(old_path,new_path):
    # shutil.copyfile(old_path, new_path)
    shutil.move(old_path,new_path)

if __name__=="__main__":
    parent_path = "old/"
    if not os.path.exists("new"):
        os.makedirs("new")

    partten = ".nii.gz"
    brainmask_list = glob.glob(parent_path+"*"+partten)# 获取所有文件路径
    suffix = "_brain_mask_MNI.nii.gz"
    for brainmask in brainmask_list:
        file_id  = os.path.basename(brainmask)#获取文件名
        file_id = file_id.replace(partten,"")#获取id
        print(file_id)
        file_copy(brainmask,"new/"+file_id+suffix)

