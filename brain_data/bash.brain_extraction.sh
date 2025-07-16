###
# make predictions on specified dataset

# Input: path to t1 head scan
# Output: path to prediction result
###

### parameters setting ###
# path to output directory
output_dir=$(echo $PWD"/brain_mask")

#echo $output_dir
#read -p "Pause L11"

# directory contains standard brain template
MNI_template_file="./MNI_standard_brain_template/MNI152_T1_1mm.nii.gz"

converted_brain_dir=$output_dir"/MNI_converted_brain/"
# directory contains extracted brain masks, in MNI space
brain_mask_MNI_dir=$output_dir"/brain_mask_MNI/"
# directory contains converted brain masks, in raw test_data space
brain_mask_raw_dir=$output_dir"/brain_mask_raw/"
raw_data_hdf5_rec_dir=$output_dir"/raw_data_hdf5_records/"
brain_mask_npy_arr_dir=$output_dir"/brain_mask_npy_arr/"

### step 0. mkdir if directories do not exist###
mkdir -p $output_dir
mkdir -p $raw_data_hdf5_rec_dir
mkdir -p $brain_mask_npy_arr_dir
mkdir -p $converted_brain_dir
mkdir -p $brain_mask_raw_dir
mkdir -p $brain_mask_MNI_dir

### step 1. get path to input head volume ###
# A POSIX variable
OPTIND=1         # Reset in case getopts has been used previously in the shell.
# Initialize our own variables:
path_to_current_t1_case=""
# get parameters from commandline
while getopts "h?f:" opt; do
    case "$opt" in
    h|\?)
        echo "Please feed path to raw T1 head volume as: -f path_to_raw_T1_volume"
        exit 0
        ;;
    f)  path_to_current_t1_case=$OPTARG
        ;;
    esac
done

shift $((OPTIND-1))

[ "${1:-}" = "--" ] && shift

echo "input_T1_file='$path_to_current_t1_case'"

if [ -z "$path_to_current_t1_case" ]; then
  echo "No argument supplied. Please provide T1 raw volume"
  exit 1
fi



### step 2. process case by case ###
### step 2.0 clean raw test_data dir (MNI converted brain) ###
echo "cleaning working directories..."
# clear MNI brain volume
rm $(echo $converted_brain_dir"*")
# clear hdf5 records
rm $(echo $raw_data_hdf5_rec_dir"*")
# clear numpy brain mask
rm $(echo $brain_mask_npy_arr_dir"*")
# clear MNI brain mask
rm $(echo $brain_mask_MNI_dir"*")

#path_to_current_t1_case="SU859_EX861_raw_t1.nii.gz"
### step 2.1 raw brain test_data ---> MNI space ###
echo "Registrating scan: "${path_to_current_t1_case}
# get base file name
base_file_name=$(basename $path_to_current_t1_case)
echo $base_file_name

# remove extension
file_name_current_case_without_extension=$(echo ${base_file_name%".nii.gz"})
 echo $file_name_current_case_without_extension

# path to transform matrix
path_to_transform_matrix=$converted_brain_dir$file_name_current_case_without_extension".mat"
# path to transformed scans
path_to_MNI_head=$converted_brain_dir$file_name_current_case_without_extension".MNI.nii.gz"

# brain registration from raw test_data space to MNI space
flirt -in $path_to_current_t1_case -ref $MNI_template_file -omat $path_to_transform_matrix -out $path_to_MNI_head -dof 6 -searchrx -20 20 -searchry -20 20 -searchrz -20 20
###
## path to inverse matrix
#path_to_inverse_transform_matrix=$converted_brain_dir$file_name_current_case_without_extension"_inverse.mat"
## generate inverse matrix
#convert_xfm -omat $path_to_inverse_transform_matrix -inverse $path_to_transform_matrix
#
#echo $path_to_MNI_head" transformed."
#
#### step 2.2 generate reference records ###
#echo "generating reference record..."
#python hdf5_generate_reference_file.py --raw_T1_file $path_to_MNI_head --hdf5_data_dir $raw_data_hdf5_rec_dir
#
#### step 2.3 inferecne ###
#echo "inferencing..."
#python main_test.py --model_dir ./reference_model --hdf5_data_dir $raw_data_hdf5_rec_dir --gpus "0, 1" --brain_mask_numpy $brain_mask_npy_arr_dir
#
#### step 2.4 numpy ---> nifti (MNI space)###
#echo "converting brain mask back to nifti format..."
#python convert_np_to_nifti.py --brain_mask_numpy $brain_mask_npy_arr_dir --brain_mask_MNI $brain_mask_MNI_dir --MNI_raw_data_dir $converted_brain_dir --hdf5_data_dir $raw_data_hdf5_rec_dir
#
#### step 2.5 MNI space ---> raw space ###
#echo "inverse brain mask from MNI space back to raw test_data space"
#path_to_brain_mask_MNI=$brain_mask_MNI_dir$file_name_current_case_without_extension".nii.gz"
#
## path to brain mask in raw space
#path_to_brain_mask_raw=$brain_mask_raw_dir$file_name_current_case_without_extension"_raw.nii.gz"
## inverse to original raw test_data space
#flirt -in $path_to_brain_mask_MNI -ref $path_to_current_t1_case -applyxfm -init $path_to_inverse_transform_matrix -out $path_to_brain_mask_raw
#
#### step 2.6 postprocessing: clear noise and fill small holes in brain mask ###
#basic_filename=${path_to_brain_mask_raw%_raw.nii.gz}
#temp_raw_bin=$basic_filename"_bin"
#fslmaths $path_to_brain_mask_raw -thr 0.5 -bin $temp_raw_bin
#
#temp_raw_cluster=$basic_filename"_cluster"
#cluster -i $temp_raw_bin -t 1 --no_table --osize=$temp_raw_cluster
#
#temp_raw_cluster_thr_bin=$basic_filename"_cluster_thr_bin"
#fslmaths $temp_raw_cluster -thr 1000000 -bin $temp_raw_cluster_thr_bin
#
#temp_raw_mask_bin=$basic_filename"_mask_bin"
#fslmaths $path_to_brain_mask_raw -mas $temp_raw_cluster_thr_bin -bin $temp_raw_mask_bin
#
#path_brain_mask=$basic_filename"_brain"
#fslmaths $temp_raw_mask_bin -fillh $path_brain_mask
#
## delete temporary files
#rm $(echo $temp_raw_bin"*")
#rm $(echo $temp_raw_cluster"*")
#rm $(echo $temp_raw_mask_bin"*")
#rm $(echo $path_to_brain_mask_raw)
#
#### output info
#echo "Done. Please find brain mask at:"
#echo $path_brain_mask".nii.gz"
#echo
