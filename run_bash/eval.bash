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
###########eval##########
#result_dir_name="result-"${datasets_name}"_"${model_name}"_"${epochs}"_"${cv}
#/mnt/SSD2/YHR/dataset/${datasets_name}/brain_class/${cv}/predict/${epochs}
python ../eval_MSD.py\
 --result_dir /mnt/HDLV1/YHR/dataset/${datasets_name}/5folder_my/${cv}/${model_name}/${epochs} \
 --test_label_dir /mnt/SSD2/YHR/dataset/${datasets_name}/5folder_my/${cv}/test_label \
 --patch_size 128 \
 --overlap_step 48 \
 --checkpoint_num ${epochs}

