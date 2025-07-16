####################参数部分
datasets_name=BraTS2023-GLI
model_name=gmsm_4x_pc
epochs=900
cv=3





python ../convert_np_nifti-BraTS2023.py \
 --result_dir /mnt/HDLV1/YHR/dataset/${datasets_name}/5folder/${cv}/${model_name}/${epochs} \
 --raw_data_dir /mnt/SSD2/YHR/dataset/BraTS2023-GLI/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
 --test_data_dir /mnt/SSD2/YHR/dataset/${datasets_name}/5folder/${cv}/test \
 --patch_size 128 \
 --overlap_step 48 \
 --checkpoint_num ${epochs}