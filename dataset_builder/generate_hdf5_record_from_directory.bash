cv=0
dataset_name=BraTS2023-GLI
##overlap_step 从8改为56
python hdf5_generator_from_directory.py \
  --raw_data_dir /mnt/SSD2/YHR/dataset/BraTS2023-GLI/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
	--output_data_dir /mnt/SSD2/YHR/dataset/${dataset_name}/5folder/${cv} \
	--patch_size 128 \
	--overlap_step 48 \
	--KFold_dir /mnt/SSD2/YHR/dataset/BraTS2023-GLI/KFold \
	--cv ${cv}

