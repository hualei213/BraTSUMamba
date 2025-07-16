####################参数部分
datasets_name="BraTS-GLI"
#model_name="UNeXt_3D_mlp_attention"
epochs=350
cv=0

###########eval##########
#result_dir_name="result-"${datasets_name}"_"${model_name}"_"${epochs}"_"${cv}
python ../evaluation-another.py\
 --result_dir /mnt/SSD2/YHR/dataset/${datasets_name}/brain_class/${cv}/predict/${epochs} \
 --test_label_dir /mnt/SSD2/YHR/dataset/${datasets_name}/brain_class/${cv}/test_label \
 --patch_size 128 \
 --overlap_step 48 \
 --checkpoint_num ${epochs}



