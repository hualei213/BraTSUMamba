####################参数部分
datasets_name="IBSR"
model_name="asmlp"
epochs=2000
cv=0
#####################
result_dir_name="result-"${model_name}"_"${datasets_name}"_"${cv}"_"${epochs}
python ../main_train_2k.py \
 --model_name ${model_name} \
 --result_dir ../实验结果整理/${datasets_name}/${result_dir_name}/ \
 --log_dir /media/sggq/MyDisk/Projects/Pytorch-UNet-2class-V1-fine-tune-dataset2k-3d+100/实验结果整理/${datasets_name}/${result_dir_name}/ \
 --patch_size 128 \
 --batch_size 4 \
 --overlap_step 48 \
 --gpus 0 \
 --lr 3e-3 \
 --epochs ${epochs} \
 --train_data_dir /media/sggq/MyDisk/datasets/${datasets_name}/brain_2class_hdf5_t1_MNI/${cv}/train/ \
 --val_data_dir /media/sggq/MyDisk/datasets/${datasets_name}/brain_2class_hdf5_t1_MNI/${cv}/val \
 --epochs_per_vali 50 \
 --mode train \

