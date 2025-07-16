#flirt -ref /mnt/HDLV1/YHR/dataset/IBSR_yhr/MNI_standard_brain_template/MNI152_T1_1mm_brain_mask.nii.gz -in /mnt/HDLV1/YHR/dataset/IBSR_yhr/raw_data/brain/IBSR_01_brain.nii.gz -omat /mnt/HDLV1/YHR/dataset/IBSR_yhr/mat_test/brian_my_affine_guess.mat


flirt -in /mnt/HDLV1/YHR/dataset/IBSR_yhr/raw_data/head_volume/IBSR_01_raw_t1.nii.gz -ref /mnt/HDLV1/YHR/dataset/IBSR_yhr/MNI_standard_brain_template/MNI152_T1_1mm.nii.gz  -omat /mnt/HDLV1/YHR/dataset/IBSR_yhr/mat_test/headvolume_t1_1mm.mat  -dof 6 -searchrx -180 180 -searchry -180 180 -searchrz -180 180
