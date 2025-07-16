####################参数部分
datasets_name="NFBS"
model_name="UNeXt3D_ablation"
epochs=2000
cv=0
#####################
result_dir_name="result-"${model_name}"_"${datasets_name}"_"${cv}"_"${epochs}
python ../convert_np_nifti.2k-3d+100.py \
 --result_dir /media/sggq/MyDisk/Projects/Pytorch-UNet-2class-V1-fine-tune-dataset2k-3d+100/实验结果整理/${datasets_name}/${result_dir_name}/ \
 --raw_data_dir /media/sggq/MyDisk/datasets/${datasets_name}/brain_mask/brain_raw_mask_MNI/ \
 --test_data_dir /media/sggq/MyDisk/datasets/${datasets_name}/brain_2class_hdf5_t1_MNI/${cv}/test/ \
 --patch_size 128 \
 --overlap_step 48 \
 --checkpoint_num ${epochs}

