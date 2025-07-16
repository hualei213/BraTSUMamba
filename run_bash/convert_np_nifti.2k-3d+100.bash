####################参数部分
datasets_name=MSD
model_name=gmsm_4x_pc
epochs=950
cv=3





python ../convert_np_nifti.2k-3d+100.py \
 --result_dir /mnt/HDLV1/YHR/dataset/${datasets_name}/5folder_my/${cv}/${model_name}/${epochs} \
 --raw_data_dir /mnt/SSD2/YHR/dataset/MSD/Task01_BrainTumour/imagesTr \
 --test_data_dir /mnt/SSD2/YHR/dataset/MSD/5folder_my/${cv}/test \
 --patch_size 128 \
 --overlap_step 48 \
 --checkpoint_num ${epochs}