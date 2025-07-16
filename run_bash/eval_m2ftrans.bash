####################参数部分
datasets_name="MSD"
#datasets_name="BraTS2023-GLI"
#model_name="sdv_tnet"
model_name="m2ftrans"
epochs=749
cv=1


###########eval##########
#result_dir_name="result-"${datasets_name}"_"${model_name}"_"${epochs}"_"${cv}
#/mnt/SSD2/YHR/dataset/${datasets_name}/brain_class/${cv}/predict/${epochs}
python ../eval_MSD.py\
 --result_dir /mnt/HDLV1/YHR/dataset/MSD/5folder/${cv}/predict_m2ftrans/${epochs} \
 --test_label_dir /mnt/SSD2/YHR/dataset/MSD/5folder/${cv}/test_label \
 --patch_size 128 \
 --overlap_step 48 \
 --checkpoint_num ${epochs}

