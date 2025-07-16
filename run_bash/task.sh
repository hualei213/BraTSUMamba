source ~/.bashrc
source /opt/app/anaconda3/bin/activate
source activate /groups/g8700001/home/u2208283086/.conda/envs/unext_3D_hr
cd /home/u2208283086/YHR/3D-unext-pc-fusion-gwj/run_bash

####################参数部分
datasets_name=BraTS2023-GLI
model_name=UNeXt3D_mlp_attention_mamba_cbam
epochs=500
cv=0


bash task-2.bash ${datasets_name} ${model_name} ${epochs} ${cv}


