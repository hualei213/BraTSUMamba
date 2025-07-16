####################参数部分
datasets_name="qin_t1_MNI152_T1_1mm"
model_name="UNeXt_3D_mlp_attention"
epochs=2000
cv=1
###########train##########
#result_dir_name="result-"${model_name}"_"${datasets_name}"_"${cv}"_"${epochs}
#python ../main_train_2k.py \
# --model_name ${model_name} \
# --result_dir ../实验结果整理/${datasets_name}/${result_dir_name}/ \
# --log_dir /mnt/SSD4T2/YHR/dataset/实验结果整理/${datasets_name}/${result_dir_name}/ \
# --patch_size 128 \
# --batch_size 4 \
# --overlap_step 48 \
# --gpus 0 \
# --lr 3e-4 \
# --epochs ${epochs} \
# --train_data_dir /mnt/SSD4T2/YHR/dataset/${datasets_name}/brain_2class_hdf5_t1_MNI/${cv}/train/ \
# --val_data_dir /mnt/SSD4T2/YHR/dataset/${datasets_name}/brain_2class_hdf5_t1_MNI/${cv}/val \
# --epochs_per_vali 50 \
# --mode train \


###########pred##########
#result_dir_name="result-"${datasets_name}"_"${model_name}"_"${epochs}"_"${cv}
#python ../main_test_2k.py \
# --model_name ${model_name} \
# --result_dir /mnt/HDLV1/YHR/dataset/synthstrip_registration/${datasets_name}/brain_class/${cv}/predict \
# --patch_size 128 \
# --overlap_step 48 \
# --batch_size 4 \
# --gpus 0 \
# --model_dir /mnt/HDLV1/YHR/UNet/unet/models/mytest \
# --mode predict \
# --checkpoint_num ${epochs} \
# --test_data_dir /mnt/HDLV1/YHR/dataset/synthstrip_registration/${datasets_name}/brain_class/${cv}/test/

#/mnt/HDLV1/YHR/dataset/synthstrip_registration/asl_epi_MNI152_T1_1mm

###########eval##########
#result_dir_name="result-"${datasets_name}"_"${model_name}"_"${epochs}"_"${cv}
#python ../evaluation-another.py\
# --result_dir /mnt/HDLV1/YHR/dataset/synthstrip_registration/${datasets_name}/brain_class/${cv}/predict \
# --test_label_dir /mnt/HDLV1/YHR/dataset/synthstrip_registration/${datasets_name}/brain_class/${cv}/test_label \
# --patch_size 128 \
# --overlap_step 48 \
# --checkpoint_num ${epochs}

#########convert############
result_dir_name="result-"${model_name}"_"${datasets_name}"_"${cv}"_"${epochs}
python ../convert_np_nifti.2k-3d+100.py \
 --result_dir /mnt/HDLV1/YHR/dataset/synthstrip_registration/${datasets_name}/brain_class/${cv}/predict \
 --raw_data_dir /mnt/HDLV1/YHR/dataset/synthstrip_registration/${datasets_name}/brain_registration/ \
 --test_data_dir /mnt/HDLV1/YHR/dataset/synthstrip_registration/${datasets_name}/brain_class/${cv}/test/ \
 --patch_size 128 \
 --overlap_step 48 \
 --checkpoint_num ${epochs}


# --result_dir
#/mnt/HDLV1/YHR/dataset/synthstrip_registration/asl_epi_MNI152_T1_1mm/brain_class/0/predict
#--raw_data_dir
#/mnt/HDLV1/YHR/dataset/synthstrip_registration/asl_epi_MNI152_T1_1mm/brain_registration
#--test_data_dir
#/mnt/HDLV1/YHR/dataset/synthstrip_registration/asl_epi_MNI152_T1_1mm/brain_class/0/test/
#--patch_size=128
#--overlap_step=48
#--checkpoint_num=2000
