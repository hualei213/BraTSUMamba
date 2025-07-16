####################参数部分
datasets_name="MSD"
#model_name="UNeXt_3D_mlp_attention"
epochs=500
cv=3


###########eval##########
#result_dir_name="result-"${datasets_name}"_"${model_name}"_"${epochs}"_"${cv}
#/mnt/SSD2/YHR/dataset/${datasets_name}/brain_class/${cv}/predict/${epochs}
python ../eval_MSD.py\
 --result_dir /mnt/HDLV1/YHR/dataset_pc/${datasets_name}/5folder/${cv}/predict/${epochs} \
 --test_label_dir /mnt/HDLV1/YHR/dataset/${datasets_name}/5folder/${cv}/test_label \
 --patch_size 128 \
 --overlap_step 48 \
 --checkpoint_num ${epochs}

