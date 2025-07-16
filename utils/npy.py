import napari

# 打开 .npy 文件
with napari.gui_qt():
    napari.view_image(np.load('/mnt/HDLV1/YHR/dataset/synthstrip_registration/asl_epi_MNI152_T1_1mm/brain_class/0/predict/test_instance-131_checkpoint_2000.npy'))

