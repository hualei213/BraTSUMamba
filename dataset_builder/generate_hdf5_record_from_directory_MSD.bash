cv=0
dataset_name=MSD
##overlap_step 从8改为56
python hdf5_generator_from_directory_MSD.py \
  --raw_data_dir /mnt/SSD2/YHR/dataset/${dataset_name}/Task01_BrainTumour/imagesTr \
	--output_data_dir /mnt/SSD2/YHR/dataset/${dataset_name}/5folder/${cv} \
	--patch_size 128 \
	--overlap_step 48 \
	--KFold_dir /mnt/SSD2/YHR/dataset/${dataset_name}/KFold \
	--cv ${cv}

