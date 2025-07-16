####################参数部分
datasets_name="IBSR"
model_name="UNeXt3D"
epochs=1300
cv=1
#####################
#result_dir_name="result-"${model_name}"_"${datasets_name}"_"${cv}"_"${epochs}
result_dir_name="result-"${model_name}"_"${datasets_name}"_"${cv}"_2000"
python ../main_test_2k.py \
 --model_name ${model_name} \
 --result_dir /media/sggq/MyDisk/Projects/Pytorch-UNet-2class-V1-fine-tune-dataset2k-3d+100/实验结果整理/${datasets_name}/${result_dir_name}/ \
 --patch_size 128 \
 --overlap_step 80 \
 --batch_size 4 \
 --gpus 0 \
 --mode predict \
 --checkpoint_num ${epochs} \
 --test_data_dir /media/sggq/MyDisk/datasets/${datasets_name}/brain_2class_hdf5_t1_MNI/${cv}/test/
