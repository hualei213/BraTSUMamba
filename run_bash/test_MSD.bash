####################参数部分
datasets_name=MSD
#datasets_name=BraTS2023-GLI
#model_name=sdv_tunet
#model_name=UNeXt_3D_mlp_attention
#model_name=UNeXt3D_mlp_attention_mamba_cbam
#model_name = unetr_pp
#model_name=UNeXt_mamba_4x_cbam_bsblock
#model_name=UNeXt_mamba_4x_cbam_bsblock
model_name=UNeXt_3D_mlp_attention_seg_counting_MSBMamba_GAB
epochs=300
cv=2



#######pred######
python ../main_test_2k.py \
 --model_name ${model_name} \
 --result_dir /mnt/HDLV1/YHR/dataset/${datasets_name}/5folder/${cv}/${model_name}/${epochs} \
 --patch_size 128 \
 --overlap_step 48 \
 --batch_size 4 \
 --gpus 0 \
 --model_dir /mnt/HDLV1/YHR/3D-unext-pc-fusion/${datasets_name}/${model_name}/${cv}/1000 \
 --mode predict \
 --checkpoint_num ${epochs} \
 --test_data_dir /mnt/SSD2/YHR/dataset/${datasets_name}/5folder_my/${cv}/test



