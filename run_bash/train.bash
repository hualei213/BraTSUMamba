####################参数部分
#datasets_name=BraTS2023-GLI
#datasets_name=BraTS2020
datasets_name=MSD
#model_name=UNeXt_3D_mlp_attention
#model_name=UNeXt_3D_mlp_attention_pc
#model_name=UNeXt_3D_mlp_attention_segmamba_6x
#model_name=UNeXt_mamba_4x_cbam_bsblock
#model_name=UNeXt_3D_mlp_gmsm_fsca_pc
#model_name=UNeXt_3D_mlp_attention_gmsm_fd_fsca_pc
#model_name=UNeXt_3D_mlp_attention_fd_gmsm_fsca_pc
#model_name=UNeXt_3D_mlp_attention_Parallelfdgmsm_fsca_pc
#model_name=gmsm_4x_scaf_pc
#model_name=gmsm_4x_scaf_pc_fusion
#model_name=gmsm_4x_scaf_pc_fdfusion
#model_name=gmsm_4x_scaf_pc_fusion_fdreweight
#model_name=gmsm_4x_scaf_pc_fusion_defreq
#model_name=gmsm_4x_pc_fusion_re
#model_name=gmsm_4x_pc
#model_name=gmsm_4x_pc_h
#model_name=gmsm_4x_pc_l
#model_name=gmsm_4x
#model_name="gmsm_4x_hlcrossatt"
#model_name="msc_2x_mamaba"
#model_name=gmsm_4x_pc_hlcrossatt
model_name="BraTSUMamba"
#model_name="enformer"
#model_name="unetr++"
epochs=1000
cv=0
###########train##########
python ../main_train_2k.py \
--model_name ${model_name} \
--result_dir /mnt/HDLV1/YHR/3D-unext-pc-fusion/${datasets_name}/${model_name}/${cv}/${epochs} \
--log_dir /mnt/SSD2/YHR/dataset/${datasets_name}/实验结果整理 \
--csv_dir /mnt/HDLV1/YHR/3D-unext-pc-fusion/${datasets_name}/${model_name}/${cv}/1000.csv \
--patch_size 128 \
--worker_num 16 \
--batch_size 4 \
--overlap_step 48 \
--gpus 0 \
--lr 3e-4 \
--epochs ${epochs} \
--train_data_dir /mnt/SSD2/YHR/dataset/${datasets_name}/5folder/${cv}/train \
--val_data_dir /mnt/SSD2/YHR/dataset/${datasets_name}/5folder/${cv}/val \
--epochs_per_vali 50 \
--mode train
