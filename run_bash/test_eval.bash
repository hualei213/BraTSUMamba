# 参数
#datasets_name="BraTS2023-GLI"
datasets_name="MSD"
#model_name="gmsm_4x_scaf_pc_fdfusion"
#model_name="gmsm_4x_pc"
#model_name="gmsm_4x_pc_h"
#model_name="gmsm_4x_pc_l"
#model_name="gmsm_4x"
#model_name="gmsm_4x_hlcrossatt"
#model_name="msc_2x_mamaba"
#model_name="gmsm_4x_pc_hlcrossatt"
model_name="BraTSUMamba"
cv=0

# 循环迭代epochs，从50开始，每次递增50直到达到1000
for epochs in {200..1000..50}
do
    echo "Running test with epochs = $epochs"

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
    --test_data_dir /mnt/SSD2/YHR/dataset/${datasets_name}/5folder/${cv}/test

    # 执行评估脚本
    python ../eval_MSD.py\
    --result_dir /mnt/HDLV1/YHR/dataset/${datasets_name}/5folder/${cv}/${model_name}/${epochs} \
    --test_label_dir /mnt/SSD2/YHR/dataset/${datasets_name}/5folder/${cv}/test_label \
    --patch_size 128 \
    --overlap_step 48 \
    --checkpoint_num ${epochs}
done