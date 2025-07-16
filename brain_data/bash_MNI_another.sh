data_path="/mnt/HDLV1/YHR/dataset/IBSR_yhr"
###
# input_path : brain_mask/brain    brain_mask/head_volume
# output_path : brain_mask/brain_raw_mask_MNI      brain_mask/brain_raw_mask_MAT
###

### parameters setting ###
# path to output directory
output_dir=$(echo ${data_path}"/brain_mask")
# directory contains standard brain template
MNI_template_file=${data_path}"/MNI_standard_brain_template/MNI152_T1_1mm.nii.gz"
# directory contains converted brain masks, in MNI test_data space
brain_raw_mask_MNI_dir=$output_dir"/brain_raw_mask_MNI/"
# directory contains mat
brain_raw_mask_MAT_dir=$output_dir"/brain_raw_mask_MAT/"
### step 0. mkdir if directories do not exist###
mkdir -p $output_dir
mkdir -p $brain_raw_mask_MNI_dir
mkdir -p $brain_raw_mask_MAT_dir

### step 1. get path to input head volume ###
# Initialize our own variables:
path_to_current_t1_case=""
# get parameters from commandline
### step 2.0 clean raw test_data dir (MNI converted brain) ###
echo "cleaning working directories..."
# clear MNI brain volume
rm $(echo $brain_raw_mask_MNI_dir"*")
rm $(echo $brain_raw_mask_MAT_dir"*")

# get fileList form folder
head_vol_folder=$(echo ${data_path}"/raw_data/head_volume")
#echo $folder
head_volume_files=$(ls $head_vol_folder)
for sfile in ${head_volume_files}
do
  ### step 2 raw head volume ---> MNI space ###
  path_to_current_t1_case=$(echo $head_vol_folder"/${sfile}")  #当前headvolume文件路径
  echo "Registrating scan: "${path_to_current_t1_case}
  # get base file name
  base_file_name=$(basename $path_to_current_t1_case)   #获取文件名
#  echo $base_file_name
  # remove extension for getting id
  file_name_current_case_without_extension=$(echo ${base_file_name%"_raw_t1.nii.gz"})
#  echo $file_name_current_case_without_extension

  # path to transform matrix
  path_to_transform_matrix=$brain_raw_mask_MAT_dir$file_name_current_case_without_extension".mat"
  # path to transformed scans
  path_to_MNI_head=$brain_raw_mask_MNI_dir$file_name_current_case_without_extension"_raw_t1_MNI.nii.gz"

  # brain registration from raw test_data space to MNI space: head volume
  echo "flirt raw_t1 to MNI"
  flirt -in $path_to_current_t1_case -ref $MNI_template_file -omat $path_to_transform_matrix -out $path_to_MNI_head -dof 6 -searchrx -180 180 -searchry -180 180 -searchrz -180 180

  # path to transformed scans
  path_to_MNI_mask=$brain_raw_mask_MNI_dir$file_name_current_case_without_extension"_brain_mask_MNI.nii.gz"
  # get brain file path
  path_to_current_t1_case2=$(echo $(echo ${data_path}"/raw_data/brain")"/${file_name_current_case_without_extension}_brain.nii.gz")

  # brain registration from brain test_data space to MNI space
  echo "flirt brain to MNI"
  # flirt -in $path_to_current_t1_case2 -ref $MNI_template_file -omat $path_to_transform_matrix -out $path_to_MNI_mask -dof 6 -searchrx -20 20 -searchry -20 20 -searchrz -20 20

  # plz refer fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT/UserGuide
  flirt -in $path_to_current_t1_case2 -ref $MNI_template_file -out $path_to_MNI_mask -init $path_to_transform_matrix -applyxfm

  ### step 2.6 postprocessing: clear noise and fill small holes in brain mask ###
  basic_filename=$brain_raw_mask_MNI_dir$file_name_current_case_without_extension  #文件路径
  temp_raw_bin=$basic_filename"_bin"
  fslmaths $path_to_MNI_mask -thr 0.5 -bin $temp_raw_bin  #0.5为阈值，大于0.5为最大值，小于0.5设为0
  path_brain_mask=$basic_filename"_brain_mask_MNI"
  fslmaths $temp_raw_bin -fillh $path_brain_mask           #填充孔洞

  # delete temporary files
  rm $(echo $temp_raw_bin"*")
  ### output info
  echo "Done. Please find brain mask at:"
  echo $path_brain_mask".nii.gz"
done