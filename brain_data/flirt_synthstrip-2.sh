data_path="/mnt/HDLV1/YHR/dataset/synthstrip_classify"
small_data_name="qin_flair"
output_path="/mnt/HDLV1/YHR/dataset/synthstrip_registration"
template_name="MNI152_T1_1mm.nii.gz"
###
# input_path : brain_mask/brain    brain_mask/head_volume
# output_path : brain_mask/brain_raw_mask_MNI      brain_mask/brain_raw_mask_MAT
###

### parameters setting ###
# path to output directory
template_name_without_extension=$(echo ${template_name%".nii.gz"})
output_dir=$output_path"/${small_data_name}_${template_name_without_extension}"
# directory contains standard brain template
MNI_template_file=${data_path}"/template/${template_name}"
# directory contains converted brain masks, in MNI test_data space
brain_registration_dir=${output_dir}"/brain_registration/"
# directory contains mat
brain_registration_MAT_dir=${output_dir}"/brain_registration_MAT/"
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
head_vol_folder=${data_path}"/${small_data_name}/image"
#echo $folder
head_volume_files=$(ls $head_vol_folder)
for sfile in ${head_volume_files}
do
  ### step 2 raw head volume ---> MNI space ###
  path_to_current_t1_case=$head_vol_folder"/${sfile}"  #当前headvolume文件路径
  echo "Registering scan: "${path_to_current_t1_case}
  # get base file name
  base_file_name=$(basename $path_to_current_t1_case)   #获取文件名
#  echo $base_file_name
  # remove extension for getting id
  file_name_current_case_without_extension=$(echo ${base_file_name%".nii.gz"})
#  echo $file_name_current_case_without_extension

  # path to transform matrix
  path_to_transform_matrix=$brain_registration_MAT_dir$file_name_current_case_without_extension".mat"
  # path to transformed scans
  path_to_brain_registration=$brain_registration_dir$file_name_current_case_without_extension".nii.gz"

  # brain registration from raw test_data space to MNI space: head volume
  echo "flirt raw_t1 to MNI"
  flirt -in $path_to_current_t1_case -ref $MNI_template_file -omat $path_to_transform_matrix -out $path_to_brain_registration -dof 6 -searchrx -180 180 -searchry -180 180 -searchrz -180 180

  # path to transformed scans
  path_to_MNI_mask=$brain_registration_dir$file_name_current_case_without_extension"_brain_mask.nii.gz"
  # get brain file path
  path_to_current_t1_case2="${data_path}/${small_data_name}/mask/${file_name_current_case_without_extension//image/mask}.nii.gz"
  # brain registration from brain test_data space to MNI space
  echo "flirt brain to MNI"
  # flirt -in $path_to_current_t1_case2 -ref $MNI_template_file -omat $path_to_transform_matrix -out $path_to_MNI_mask -dof 6 -searchrx -20 20 -searchry -20 20 -searchrz -20 20

  # plz refer fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT/UserGuide
  flirt -in $path_to_current_t1_case2 -ref $MNI_template_file -out $path_to_MNI_mask -init $path_to_transform_matrix -applyxfm

  ### step 2.6 postprocessing: clear noise and fill small holes in brain mask ###
  basic_filename="$brain_registration_dir${file_name_current_case_without_extension}" #文件路径
  temp_raw_bin=$basic_filename"_bin"
  fslmaths $path_to_MNI_mask -thr 0.5 -bin $temp_raw_bin  #0.5为阈值，大于0.5为最大值，小于0.5设为0
  path_brain_mask=$basic_filename"_brain_mask"
  fslmaths $temp_raw_bin -fillh $path_brain_mask           #填充孔洞

  # delete temporary files
  rm $(echo $temp_raw_bin"*")
  ### output info
  echo "Done. Please find brain mask at:"
  echo $path_brain_mask".nii.gz"
done