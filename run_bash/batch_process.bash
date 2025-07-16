####################参数部分
datasets_name="synthstrip_merge_5folder_182"
#model_name="UNeXt_3D_mlp_attention"
epochs=500
cv=0
############train##########
#python ../main_train_2k.py \
#--model_name UNeXt_3D_mlp_attention \
#--result_dir /mnt/HDLV1/YHR/UNet/实验结果整理/${cv}/seg_500 \
#--model-dir_reg /mnt/HDLV1/YHR/UNet/实验结果整理/${cv}/reg_500 \
#--log_dir /mnt/HDLV1/YHR/dataset/实验结果整理 \
#--csv_dir /mnt/HDLV1/YHR/UNet/实验结果整理/${cv}/model_reg.csv \
#--patch_size 128 \
#--batch_size 2 \
#--overlap_step 48 \
#--gpus 0 \
#--lr 3e-4 \
#--epochs ${epochs} \
#--train_data_dir /mnt/HDLV1/YHR/dataset/${datasets_name}/brain_class/${cv}/train \
#--val_data_dir /mnt/HDLV1/YHR/dataset/${datasets_name}/brain_class/${cv}/val \
#--epochs_per_vali 50 \
#--mode train \
#--atlas /mnt/HDLV1/YHR/dataset/${datasets_name}/brain_class/${cv}/atlas \
#--flirtReg_dir /mnt/HDLV1/YHR/dataset/synthstrip_flirt_reg_182_hdf5 \








#result_dir_name="result-"${model_name}"_"${datasets_name}"_"${cv}"_"${epochs}
#python ../main_train_2k.py \
# --model_name ${model_name} \
# --result_dir ../实验结果整理/${datasets_name}/${result_dir_name}/ \
# --log_dir /mnt/HDLV1/YHR/dataset/实验结果整理/${datasets_name}/${result_dir_name}/ \
# --patch_size 128 \
# --batch_size 4 \
# --overlap_step 48 \
# --gpus 0 \
# --lr 3e-4 \
# --epochs ${epochs} \
# --train_data_dir /mnt/HDLV1/YHR/dataset/${datasets_name}/brain_class/${cv}/train \
# --val_data_dir /mnt/HDLV1/YHR/dataset/${datasets_name}/brain_class/${cv}/val \
# --epochs_per_vali 50 \
# --mode train \


###########pred##########
#result_dir_name="result-"${datasets_name}"_"${model_name}"_"${epochs}"_"${cv}
#python ../main_test_2k.py \
# --model_name ${model_name} \
# --result_dir /mnt/HDLV1/YHR/dataset/${datasets_name}/brain_class/${cv}/predict/${epochs} \
# --patch_size 128 \
# --overlap_step 48 \
# --batch_size 4 \
# --gpus 0 \
# --model_dir /mnt/HDLV1/YHR/UNet/unet/models/model_yhr \
# --mode predict \
# --checkpoint_num ${epochs} \
# --test_data_dir /mnt/HDLV1/YHR/dataset/${datasets_name}/brain_class/${cv}/test

#/mnt/HDLV1/YHR/dataset/synthstrip_registration/asl_epi_MNI152_T1_1mm

###########eval##########
#result_dir_name="result-"${datasets_name}"_"${model_name}"_"${epochs}"_"${cv}
#python ../evaluation-another.py\
# --result_dir /mnt/HDLV1/YHR/dataset/${datasets_name}/brain_class/${cv}/predict/${epochs} \
# --test_label_dir /mnt/HDLV1/YHR/dataset/${datasets_name}/brain_class/${cv}/test_label \
# --patch_size 128 \
# --overlap_step 48 \
# --checkpoint_num ${epochs}


#########convert############
#result_dir_name="result-"${model_name}"_"${datasets_name}"_"${cv}"_"${epochs}
#python ../convert_np_nifti.2k-3d+100.py \
# --result_dir /media/sggq/MyDisk/Projects/Pytorch-UNet-2class-V1-fine-tune-dataset2k-3d+100/实验结果整理/${datasets_name}/${result_dir_name}/ \
# --raw_data_dir /media/sggq/MyDisk/datasets/${datasets_name}/brain_mask/brain_raw_mask_MNI/ \
# --test_data_dir /media/sggq/MyDisk/datasets/${datasets_name}/brain_2class_hdf5_t1_MNI/${cv}/test/ \
# --patch_size 128 \
# --overlap_step 48 \
# --checkpoint_num ${epochs}


#result_dir_name="result-"${model_name}"_"${datasets_name}"_"${cv}"_"${epochs}
python ../convert_np_nifti.2k-3d+100.py \
 --result_dir /mnt/HDLV1/YHR/dataset/${datasets_name}/brain_class/${cv}/predict/${epochs} \
 --raw_data_dir /mnt/HDLV1/YHR/dataset/synthstrip_merge_resize_182 \
 --test_data_dir /mnt/HDLV1/YHR/dataset/${datasets_name}/brain_class/${cv}/test \
 --patch_size 128 \
 --overlap_step 48 \
 --checkpoint_num ${epochs}







# --result_dir
#/mnt/HDLV1/YHR/dataset/synthstrip_registration/qin_t2_MNI152_T1_1mm/brain_class/1/predict
#--raw_data_dir
#/mnt/HDLV1/YHR/dataset/synthstrip_registration/qin_t2_MNI152_T1_1mm/brain_registration
#--test_data_dir
#/mnt/HDLV1/YHR/dataset/synthstrip_registration/qin_t2_MNI152_T1_1mm/brain_class/1/test
#--patch_size=128
#--overlap_step=48
#--checkpoint_num=2000
