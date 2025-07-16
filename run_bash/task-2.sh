####################参数部分
datasets_name=${1}
#datasets_name=MSD
model_name=${2}
#model_name="enformer"
epochs=${3}
cv=${4}
###########train##########
python ../main_train_2k.py \
--model_name ${model_name} \
--result_dir /mnt/HDLV1/YHR/3D-unext-pc-fusion/${datasets_name}/${model_name}_4x/${cv}/${epochs} \
--log_dir /mnt/SSD2/YHR/dataset/${datasets_name}/实验结果整理 \
--csv_dir /mnt/HDLV1/YHR/3D-unext-pc-fusion/${datasets_name}/${model_name}_4x/${cv}/1000.csv \
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



#######pred######
python ../main_test_2k.py \
 --model_name ${model_name} \
 --result_dir /mnt/HDLV1/YHR/dataset/${datasets_name}/5folder_my/${cv}/${model_name}_4x/${epochs} \
 --patch_size 128 \
 --overlap_step 48 \
 --batch_size 4 \
 --gpus 0 \
 --model_dir /mnt/HDLV1/YHR/3D-unext-pc-fusion/${datasets_name}/${model_name}_4x/${cv}/1000 \
 --mode predict \
 --checkpoint_num ${epochs} \
 --test_data_dir /mnt/SSD2/YHR/dataset/${datasets_name}/5folder_my/${cv}/test



 python ../eval_MSD.py\
 --result_dir /mnt/HDLV1/YHR/dataset/${datasets_name}/5folder/${cv}/${model_name}_4x/${epochs} \
 --test_label_dir /mnt/SSD2/YHR/dataset/${datasets_name}/5folder/${cv}/test_label \
 --patch_size 128 \
 --overlap_step 48 \
 --checkpoint_num ${epochs}


