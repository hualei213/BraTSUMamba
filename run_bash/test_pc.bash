####################参数部分
datasets_name="MSD"
model_name="UNeXt_3D_mlp_attention"
epochs=500
cv=3



#######pred######
python ../main_test.py \
 --model_name ${model_name} \
 --result_dir /mnt/HDLV1/YHR/dataset_pc/${datasets_name}/5folder/${cv}/predict/${epochs} \
 --patch_size 128 \
 --overlap_step 48 \
 --batch_size 4 \
 --gpus 0 \
 --model_dir /mnt/HDLV1/YHR/3D-unext-pc-ys/MSD-test/3-500 \
 --mode predict \
 --checkpoint_num ${epochs} \
 --test_data_dir /mnt/HDLV1/YHR/dataset/${datasets_name}/5folder/${cv}/test