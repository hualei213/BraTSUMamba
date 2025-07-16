import argparse
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.utils import AverageMeter
from torch.utils.data import DataLoader
from torchvision import transforms
import time

from dataset_builder import BrainDatasetTest, TestToTensor
from main_test_2k import get_case_id_list
from unet.Modified3DUNet import Modified3DUNet
from unet.UNeXt_3D import UNeXt3D
from unet.unet3D import _3DUNet


def parse_args():
    data_set = "IBSR"
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=data_set+"_UNext_woDS",
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    print(torch.get_num_threads())
    torch.set_num_threads(20)
    torch.set_flush_denormal(True)
    print(torch.get_num_threads())
    ####################参数部分
    overlap_step = 48
    patch_size=128
    epoch = 2000
    dataset_name = "IBSR"
    model_name="UNeXt"
    # model_name="ModifyUnet"
    cv=0
    test_data_dir = "/media/shenhualei/SSD05/sggq/projects/Pytorch-UNet-2class-V1-fine-tune-dataset2k-3d+100/" \
                    "实验结果整理/"+dataset_name+"/result-"+model_name+"_"+dataset_name+"_"+str(cv)+"_"+str(epoch)+"/"
    test_hdf5_file_path = "/media/shenhualei/SSD05/sggq/datasets/"+dataset_name+"/brain_2class_hdf5_t1_MNI/"+str(cv)+"/test/"
    modelpath=os.path.join(test_data_dir,"model-"+str(epoch))
    model = UNeXt3D(1, 2)
    # model = Modified3DUNet(n_channels=1, n_classes=2)
    # model = _3DUNet(in_channels=1, num_classes=2, batch_normal=True, bilinear=False)
    ###################
    # cudnn.benchmark = True
    gput = AverageMeter()
    cput = AverageMeter()
    pattern = '*-test-overlap-%d-patch-%d.hdf5' % (overlap_step, patch_size)
    test_list = get_case_id_list(test_hdf5_file_path, pattern)
    count = 0
    model = model.to(torch.device("cuda"))
    #### 加载模型
    model.load_state_dict(torch.load(modelpath))
    model.eval()
    with torch.no_grad():
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
                                          num_workers=0)
            # compute output
            if count<=1:
                start = time.time()
                for sample in dataloaders_test:
                    input = sample["image"]
                    input = input.to(torch.device("cuda"))
                    model = model.to(torch.device("cuda"))
                    # get prediction id
                    print('-' * 20 + 'Load ' + str(test_instance_id) + '-' * 20)
                    # get input patches
                    input_images = input
                    # get ids of input patches
                    patch_id = sample['patch_id']
                    patch_id = patch_id.type(torch.int16)
                    patch_id = torch.squeeze(patch_id, dim=0)

                    original_shape = sample['original_shape'].numpy()
                    original_shape = np.squeeze(original_shape)
                    batch_group_mile_stone = list(range(0, patch_id.size(0), 1))
                    if patch_id.size(0) % 1 != 0:
                        # add id of the final patch
                        batch_group_mile_stone.append(patch_id.size(0))

                    for i in range(len(batch_group_mile_stone)):
                        # skip the final patch id
                        if i == len(batch_group_mile_stone) - 1:
                            pass
                        else:
                            # print('Evaluate %dth batch, total %d'
                            #       % (i + 1, len(batch_group_mile_stone)))
                            # get input patches
                            input_patches = [input_images[:, :, :, :,
                                             j * patch_size:(j + 1) * patch_size]
                                             for j in list(range(batch_group_mile_stone[i],
                                                                 batch_group_mile_stone[i + 1]))]
                            # get id of input patches
                            input_patches_id = [patch_id[j, :].numpy()
                                                for j in list(range(batch_group_mile_stone[i],
                                                                    batch_group_mile_stone[i + 1]))]

                            input_patches = torch.stack(input_patches)
                            input_patches = torch.squeeze(input_patches, 1)
                            # predict
                            # output_patches = model(input_patches)
                stop = time.time()
                torch.cuda.empty_cache()
                gput.update(stop-start, input.size(0))
                #######################################################
                for sample in dataloaders_test:
                    input = sample["image"]
                    input = input.to(torch.device("cpu"))
                    model = model.to(torch.device("cpu"))
                    # get prediction id
                    print('-' * 20 + 'Load ' + str("CPU") + '-' * 20)
                    # get input patches
                    input_images = input
                    # get ids of input patches
                    patch_id = sample['patch_id']
                    patch_id = patch_id.type(torch.int16)
                    patch_id = torch.squeeze(patch_id, dim=0)

                    original_shape = sample['original_shape'].numpy()
                    original_shape = np.squeeze(original_shape)
                    batch_group_mile_stone = list(range(0, patch_id.size(0), 1))
                    if patch_id.size(0) % 1 != 0:
                        # add id of the final patch
                        batch_group_mile_stone.append(patch_id.size(0))
                    total_time = 0.0
                    for i in range(len(batch_group_mile_stone)):
                        # skip the final patch id
                        if i == len(batch_group_mile_stone) - 1:
                            pass
                        else:
                            # print('Evaluate %dth batch, total %d'
                            #       % (i + 1, len(batch_group_mile_stone)))
                            # get input patches
                            input_patches = [input_images[:, :, :, :,
                                             j * patch_size:(j + 1) * patch_size]
                                             for j in list(range(batch_group_mile_stone[i],
                                                                 batch_group_mile_stone[i + 1]))]
                            # get id of input patches
                            input_patches_id = [patch_id[j, :].numpy()
                                                for j in list(range(batch_group_mile_stone[i],
                                                                    batch_group_mile_stone[i + 1]))]

                            input_patches = torch.stack(input_patches)
                            input_patches = torch.squeeze(input_patches, 1)
                            # predict
                            start = time.time()
                            output_patches = model(input_patches)
                            stop = time.time()
                            total_time = total_time +(stop-start)
                cput.update(total_time, input.size(0))
                count=count+1
        print('CPU: %.4f' %cput.avg)
        print('GPU: %.4f' %gput.avg)
        # val_path = 'models/' + config['name'] + '/inference.csv'
        # f = open(val_path, "w")
        # f.write("CPU,GPU\n")
        # f.write(str(cput.avg) + "," + str(gput.avg) + "\n")
        # f.close()


if __name__ == '__main__':
    main()
