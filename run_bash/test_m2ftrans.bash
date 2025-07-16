####################参数部分
datasets_name="MSD"
#datasets_name="BraTS2023-GLI"
#model_name="sdv_tunet"
model_name="m2ftrans"
epochs=749
cv=1



#######pred######
python ../main_test_2k.py \
 --model_name ${model_name} \
 --result_dir /mnt/HDLV1/YHR/dataset/MSD/5folder/${cv}/predict_m2ftrans/${epochs} \
 --patch_size 128 \
 --overlap_step 48 \
 --batch_size 1 \
 --gpus 0 \
 --model_dir /mnt/HDLV1/YHR/3D-unext-pc-fusion/m2trasns_MSD_test/4-749 \
 --mode predict \
 --checkpoint_num ${epochs} \
 --test_data_dir /mnt/SSD2/YHR/dataset/MSD/5folder/1/test