####################参数部分
datasets_name="IBSR"
model_name="UNeXt3D"
epochs=1300
cv=1
#####################
#result_dir_name="result-"${model_name}"_"${datasets_name}"_"${cv}"_"${epochs}
result_dir_name="result-"${model_name}"_"${datasets_name}"_"${cv}"_2000"
python ../evaluation-another.py\
 --result_dir ../实验结果整理/${datasets_name}/${result_dir_name}/ \
 --test_label_dir /media/sggq/MyDisk/datasets/${datasets_name}/brain_2class_hdf5_t1_MNI/${cv}/test_label/ \
 --patch_size 128 \
 --overlap_step 80 \
 --checkpoint_num ${epochs}

