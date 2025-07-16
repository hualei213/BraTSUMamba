import  subprocess

# 定义FLIRT命令和参数
flirt_command = '/home/shl/fsl/bin/flirt'
in_file = '-in /mnt/HDLV1/YHR/dataset/synthstrip_merge/asl_epi_101_image.nii.gz'
reference_file = '-ref /mnt/HDLV1/YHR/dataset/synthstrip_classify/template/MNI152_T1_1mm.nii.gz'
out_file = '-out /mnt/HDLV1/YHR/dataset/subprocess_test/asl_epi_105.nii.gz'
omat_file = '-omat /mnt/HDLV1/YHR/dataset/subprocess_test/transformation.mat'

command = f"{flirt_command} {in_file} {reference_file} {out_file} {omat_file}"

result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# # 打印命令输出
# print(result.stdout.decode())


# def flirt_syn():
