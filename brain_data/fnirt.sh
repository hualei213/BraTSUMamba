data_path="/mnt/HDLV1/YHR/dataset/IBSR_yhr"
config_path="/mnt/HDLV1/YHR/UNet/brain_data/T1_2_MNI152_2mm.cnf"

output_dir="${data_path}/fnirt_brainout"
mkdir -p $output_dir

# get fileList form folder
head_vol_folder="${data_path}/raw_data/head_volume"
path_to_current_t1_case=""
head_volume_files=$(ls "$head_vol_folder")
for sfile in $head_volume_files;
do
#  echo "$sfile"
  path_to_current_t1_case="${head_vol_folder}/${sfile}"
  echo "Registrating scan: "${path_to_current_t1_case}
  # get base file name
  base_file_name=$(basename $path_to_current_t1_case)
  # remove extension for getting id
  file_name_current_case_without_extension="${base_file_name%_raw_t1.nii.gz}"

  path_to_current_t1_case2="${data_path}/raw_data/brain/${file_name_current_case_without_extension}_brain.nii.gz"
  #get mat
  file_name_with_mat="${file_name_current_case_without_extension}.mat"
  echo "flirt raw_t1 to MNI"
  fnirt --in=${path_to_current_t1_case} --inmask=${path_to_current_t1_case2} --aff=${file_name_with_mat} --iout=$a{output_dir} --ref=/mnt/HDLV1/YHR/dataset/IBSR_yhr/MNI_standard_brain_template/MNI152_T1_1mm.nii.gz --refmask=/mnt/HDLV1/YHR/dataset/IBSR_yhr/MNI_standard_brain_template/MNI152_T1_1mm.nii.gz
done


#--imprefm=1 --impinm=1 --imprefval=0 --impinval=0 --subsamp=4,2,1 --miter=5,5,10 --infwhm=10,6,2 --reffwhm=8,4,0 --lambda=300,75,30 --estint=1,1,0 --applyrefmask=0,0,1 --applyinmask=0,0,1 --warpres=10,10,10 --ssqlambda=1 --regmod=bending_energy --intmod=global_non_linear_with_bias --intorder=5 --biasres=50,50,50 --biaslambda=10000 --refderiv=0