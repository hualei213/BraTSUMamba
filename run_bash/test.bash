####################参数部分
#datasets_name=BraTS2023-GLI
#datasets_name=BraTS2020
datasets_name=MSD
#model_name=UNeXt_3D_mlp_attention
#model_name=UNeXt_3D_mlp_attention_pc
#model_name=UNeXt_3D_mlp_attention_segmamba_6x
#model_name=UNeXt_3D_mlp_attention_pc_gmsm_fsca
#model_name=UNeXt_3D_mlp_attention_gmsm_fd_fsca_pc
#model_name=UNeXt_3D_mlp_attention_Parallelfdgmsm_fsca_pc
#model_name=gmsm_4x_scaf_pc
#model_name=gmsm_4x_scaf_pc_fusion
model_name=gmsm_4x_scaf_pc_fdfusion
#model_name=UNeXt_3D_mlp_attention_fd_gmsm_fsca_pc
#model_name=UNeXt_mamba_4x_cbam_bsblock
#model_name="enformer"
epochs=849
cv=3

#######pred######
python ../main_test_2k.py \
 --model_name ${model_name} \
 --result_dir /mnt/HDLV1/YHR/dataset/${datasets_name}/5folder_my/${cv}/${model_name}/${epochs} \
 --patch_size 128 \
 --overlap_step 48 \
 --batch_size 4 \
 --gpus 0 \
 --model_dir /mnt/HDLV1/YHR/3D-unext-pc-fusion/${datasets_name}/${model_name}/${cv}/1000 \
 --mode predict \
 --checkpoint_num ${epochs} \
 --test_data_dir /mnt/SSD2/YHR/dataset/${datasets_name}/5folder_my/${cv}/test


# --model_name=MSD
# --result_dir=/mnt/HDLV1/YHR/dataset/MSD/5folder_vision
# --patch_size=128
# --overlap_step=48
# --batch_size=4
# --gpus=0
# --model_dir=/mnt/HDLV1/YHR/3D-unext-pc-fusion/MSD/gmsm_4x_hlcrossatt/3/850
# --mode=predict
# --checkpoint_num=850
# --test_data_dir=/mnt/SSD2/YHR/dataset/MSD/5folder_my/3/test
 BRATS_451