output_dir=$(echo $PWD"/brain_mask")
brain_mask_MNI_dir=$output_dir"/brain_mask_MNI/"
converted_brain_dir=$output_dir"/brain_raw_mask_MAT/"
# directory contains converted brain masks, in raw test_data space
brain_mask_raw_dir=$output_dir"/brain_mask_raw/"
head_vol_folder=$(echo $PWD"/raw_data/head_volume/")

mkdir -p $brain_mask_raw_dir
rm $(echo $brain_mask_raw_dir"*")

brain_mask_files=$(ls $brain_mask_MNI_dir)
for sfile in ${brain_mask_files}
do
  path_to_current_mask_case=${sfile}
  ### step 2.1 raw brain test_data ---> MNI space ###
  echo "Registrating scan: "${path_to_current_mask_case}
  # get base file name
  base_file_name=$(basename $path_to_current_mask_case)
#  echo $base_file_name


  # remove extension
  file_name_current_case_without_extension=$(echo ${base_file_name%"_predict_I_brain.nii.gz"})
  echo $file_name_current_case_without_extension
  ### step 2.5 MNI space ---> raw space ###
  echo "inverse brain mask from MNI space back to raw test_data space"
  path_to_brain_mask_MNI=$brain_mask_MNI_dir$file_name_current_case_without_extension"_predict_I_brain.nii.gz"
  path_to_current_t1_case=$head_vol_folder$file_name_current_case_without_extension"_raw_t1.nii.gz"

  # path to inverse matrix
  path_to_transform_matrix=$converted_brain_dir$file_name_current_case_without_extension".mat"
  path_to_inverse_transform_matrix=$converted_brain_dir$file_name_current_case_without_extension"_inverse.mat"

  # path to brain mask in raw space
  path_to_brain_mask_raw=$brain_mask_raw_dir$file_name_current_case_without_extension"_brain.nii.gz"
  # inverse to original raw test_data space
  ##
  # path to inverse matrix
  path_to_inverse_transform_matrix=$converted_brain_dir$file_name_current_case_without_extension"_inverse.mat"
  # generate inverse matrix
  convert_xfm -omat $path_to_inverse_transform_matrix -inverse $path_to_transform_matrix

  flirt -in $path_to_brain_mask_MNI -ref $path_to_current_t1_case -applyxfm -init $path_to_inverse_transform_matrix -out $path_to_brain_mask_raw

  ### step 2.6 postprocessing: clear noise and fill small holes in brain mask ###
  basic_filename=${path_to_brain_mask_raw%_brain.nii.gz}
  temp_raw_bin=$basic_filename"_bin"
  fslmaths $path_to_brain_mask_raw -thr 0.5 -bin $temp_raw_bin
  path_brain_mask=$basic_filename
  fslmaths $temp_raw_bin -fillh $path_brain_mask

  # delete temporary files
  rm $(echo $temp_raw_bin"*")
  ### output info
  echo "Done. Please find brain mask at:"
  echo $path_brain_mask".nii.gz"
done


