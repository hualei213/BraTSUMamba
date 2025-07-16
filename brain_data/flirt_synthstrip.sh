data_path="/mnt/SSD2/YHR/dataset/BraTS-GLI"
small_data_name="BraTS2023-GLI-Training"
output_path="/mnt/SSD2/YHR/dataset/BraTS-GLI"
#template_name="MNI152_T1_1mm_resize.nii.gz"
###
# input_path : brain_mask/brain    brain_mask/head_volume
# output_path : brain_mask/brain_raw_mask_MNI      brain_mask/brain_raw_mask_MAT
###

### parameters setting ###
# path to output directory
#template_name_without_extension=$(echo ${template_name%".nii.gz"})
#output_dir=$output_path"/${small_data_name}_${template_name_without_extension}"
# directory contains standard brain template
MNI_template_file="/mnt/SSD2/YHR/dataset/template/MNI152_T1_1mm_255/MNI152_T1_1mm_resize.nii.gz"
# directory contains converted brain masks, in MNI test_data space
brain_registration_dir=${output_path}"/BraTS2023-GLI-Traning-flirt"
# directory contains mat
brain_registration_MAT_dir=${output_path}"/brain_registration_MAT"
### step 0. mkdir if directories do not exist###
mkdir -p $output_dir
mkdir -p $brain_registration_dir
mkdir -p $brain_registration_MAT_dir

### step 1. get path to input head volume ###
# Initialize our own variables:
path_to_current_t1_case=""
# get parameters from commandline
### step 2.0 clean raw test_data dir (MNI converted brain) ###
echo "cleaning working directories..."
# clear MNI brain volume
#rm $(echo $brain_registration_dir"*")
#rm $(echo $brain_registration_MAT_dir"*")

# get fileList form folder
head_vol_folder=${data_path}"/${small_data_name}"
#echo $folder
#head_volume_files=$(ls $head_vol_folder)
#head_volume_files=$(ls "$head_vol_folder" | grep "image")
#head_volume_files=$(ls "$head_vol_folder")
head_volume_dirs=$(ls -d "$head_vol_folder")
for sfile_folder in ${head_volume_files}
do
  for sfile in $(ls "$sfile_folder" | grep "t")
  do
    ### step 2 raw head volume ---> MNI space ###
    path_to_current_t1_case=$head_vol_folder"/${sfile_folder}/${sfile}"  #当前headvolume文件路径
    echo "Registering scan: "${path_to_current_t1_case}
    # get base file name
    base_file_name=$(basename $path_to_current_t1_case)   #获取文件名
    #echo $base_file_name
    # remove extension for getting id
    file_name_current_case_without_extension=$(echo ${base_file_name%".nii.gz"})
    #echo $file_name_current_case_without_extension

    #path to transform matrix
    path_to_transform_matrix=$brain_registration_MAT_dir$file_name_current_case_without_extension".mat"
    #path to transformed scans
    path_to_brain_registration=$brain_registration_dir$file_name_current_case_without_extension".nii.gz"

    # brain registration from raw test_data space to MNI space: head volume
    echo "flirt raw_t1 to MNI"
    flirt -in $path_to_current_t1_case -ref $MNI_template_file -omat $path_to_transform_matrix -out $path_to_brain_registration -dof 6 -searchrx -180 180 -searchry -180 180 -searchrz -180 180
  done
done