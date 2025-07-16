cv=3
dataset_name=BraTS2020
##overlap_step 从8改为56
python hdf5_generator_from_directory_BraTS2020.py \
  --raw_data_dir /mnt/SSD2/YHR/dataset/${dataset_name}/MICCAI_BraTS2020_TrainingData \
	--output_data_dir /mnt/SSD2/YHR/dataset/${dataset_name}/5folder/${cv} \
	--patch_size 128 \
	--overlap_step 48 \
	--KFold_dir /mnt/SSD2/YHR/dataset/${dataset_name}/KFold \
	--cv ${cv}


