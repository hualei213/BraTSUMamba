import argparse
import os

from unet.Modified3DUNet import Modified3DUNet
from unet.unet3D import _3DUNet

os.environ["CUDA_VISIBLE_DEVICES"]="-1"# Disable GPU
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.utils import AverageMeter
from torch.utils.data import DataLoader
from torchvision import transforms
import time

from dataset_builder import BrainDatasetTest, TestToTensor
from main_test_2k import get_case_id_list
from unet.UNeXt_3D import UNeXt3D



def main():
    torch.set_num_threads(16)
    os.environ["OMP_NUM_THREADS"]="16"
    os.environ["OPENBLAS_NUM_THREADS"]="16"
    os.environ["MKL_NUM_THREADS"]="16"
    os.environ["VECLIB_MAXIMUM_THREADS"]="16"
    os.environ["NUMEXPR_NUM_THREADS"]="16"
    torch.set_flush_denormal(True)
    print(torch.get_num_threads())
    ####################参数部分
    data_set_name = "NFBS"
    overlap_step = 48
    patch_size=128
    cv = 1
    epoch=300
    test_data_dir = "/media/sggq/MyDisk/Projects/Pytorch-UNet-2class-V1-fine-tune-dataset2k-3d+100/result/model/modified/"+data_set_name+"/"+str(cv)
    test_hdf5_file_path = "/media/sggq/MyDisk/datasets/"+data_set_name+"/brain_2class_hdf5_t1_MNI/"+str(cv)+"/test/"
    modelpath=os.path.join(test_data_dir,"model-"+str(epoch))
    ###################
    cudnn.benchmark = True
    cput = AverageMeter()
    pattern = '*-test-overlap-%d-patch-%d.hdf5' % (overlap_step, patch_size)
    test_list = get_case_id_list(test_hdf5_file_path, pattern)
    sample_count = 0
    # with torch.no_grad():
    with torch.autograd.set_grad_enabled(False):
        model = _3DUNet(1,2,True,False)
        # model = Modified3DUNet(1,2)
        #### 加载模型
        model.load_state_dict(torch.load(modelpath,map_location=torch.device('cpu')),)
        print("load model: %s"%modelpath)
        model.eval()

        val_path = os.path.join(test_data_dir, "inference.txt")
        f = open(val_path, "w")
        for test_instance_id in test_list:
            test_dataset = BrainDatasetTest(root_dir=test_hdf5_file_path,
                                            test_instance_id=test_instance_id,
                                            patch_size=patch_size,
                                            overlap_step=overlap_step,
                                            transform=transforms.Compose([
                                                TestToTensor()
                                            ]))
            # build test dataloader
            dataloaders_test = DataLoader(test_dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=16)
            # compute output
            if sample_count <= 1000:
                print('Start inferencing %d-th sample' % sample_count)
                for sample in dataloaders_test:
                    input = sample["image"]
                    # get prediction id
                    print('-' * 20 + 'Load ' + str("CPU") + '-' * 20)
                    # get input patches
                    input_images = input
                    # get ids of input patches
                    patch_id = sample['patch_id']
                    patch_id = patch_id.type(torch.int16)
                    patch_id = torch.squeeze(patch_id, dim=0)
                    batch_group_mile_stone = list(range(0, patch_id.size(0), 1))
                    if patch_id.size(0) % 1 != 0:
                        # add id of the final patch
                        batch_group_mile_stone.append(patch_id.size(0))
                    total_time=0.0
                    for i in range(len(batch_group_mile_stone)):
                        # skip the final patch id
                        if i == len(batch_group_mile_stone) - 1:
                            pass
                        else:
                            input_patches = [input_images[:, :, :, :,
                                             j * patch_size:(j + 1) * patch_size]
                                             for j in list(range(batch_group_mile_stone[i],
                                                                 batch_group_mile_stone[i + 1]))]
                            input_patches = torch.stack(input_patches)
                            input_patches = torch.squeeze(input_patches, 1)
                            # predict
                            start = time.time()
                            output_patches = model(input_patches)
                            stop = time.time()
                            print('The %d-th patch done!'%i)
                            total_time=total_time+(stop-start)
                    print('-' * 30)
                    print('%d-th sample, total time: %f s'%(sample_count,total_time))
                    f.write('%d-th sample, total time: %f s'%(sample_count,total_time))
                    cput.update(total_time, input.size(0))
                sample_count = sample_count + 1
        print('CPU: %.4f' %cput.avg)
        f.write("CPU inference avg time："+str(cput.avg))
        f.close()


if __name__ == '__main__':
    main()
