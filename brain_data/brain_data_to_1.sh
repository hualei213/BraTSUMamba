#### 参数部分
## 待处理的文件路径
input_data_path=/media/sggq/MyDisk/实验结果/models/NFBS/MSMHA_CNN_4/dataset_1k/
## 待处理的文件后缀
file_suffix="_predict_I_brain.nii.gz"
## 处理后的输出文件路径，默认和待处理的文件相同路径，且重名（文件覆盖）
output_path=$input_data_path
####

###程序部分
## step：1找到待处理文件路径
find ${input_data_path} -name "*"${file_suffix} | while read file_path; do
  # 获取待处理文件路径
  echo "Registrating scan: "$file_path
  ## step：2 获取文件名
  base_file_name=$(basename $file_path)
  ## step：3 remove file suffix
  file_name_current_case_without_extension=$(echo ${base_file_name%${file_suffix}})
#  echo $file_name_current_case_without_extension
  # output file path formate same input data file name
  ## step:3 generate binary class file from input file
  temp_raw_bin=$file_name_current_case_without_extension"_bin"
  fslmaths $file_path -thr 0.5 -bin $temp_raw_bin

  path_brain_mask=$output_path$file_name_current_case_without_extension${file_suffix%".nii.gz"}
  fslmaths $temp_raw_bin -fillh $path_brain_mask

  # delete temporary files
  rm $(echo $temp_raw_bin"*")
  ### output info
  echo "Done. output path: "$path_brain_mask".nii.gz"
done