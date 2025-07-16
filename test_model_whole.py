import time
import torch.autograd
import torch.cuda
import torch.nn as nn
import torch.nn.functional
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import itertools
import torch.nn.functional as F
import torch

from dataset_builder import BrainDataset, BrainDatasetTest, RandomCrop, TestToTensor, ToTensor
from dataset_builder import BrainDataset_MSD,BrainDatasetTest_MSD,RandomCrop_MSD,TestToTensor_MSD,ToTensor_MSD
from config import get_args

import pdb

#####################################################################################
# set parameters

args = get_args()
print(args)
### parameters:
# Directory of train test_data
train_data_dir = args.train_data_dir
# Directory of validation test_data
val_data_dir = args.val_data_dir
# Directory of test test_data
test_data_dir = args.test_data_dir
# test instance id
test_instance_id = args.test_instance_id
# Patch size
patch_size = args.patch_size
batch_size = args.batch_size
num_works = args.worker_num
num_classes = args.class_num


####################################################################################
def test_model(model,
               device=torch.device("cpu"),
               test_instance_id=None):
    """
    Make prediction with the model on the entire image without patching.
    :param model: the trained model
    :param device: device to use (CPU or GPU)
    :param test_instance_id: ID of the test instance
    :return: None, saves the result as a .npy file
    """
    torch.backends.cudnn.benchmark = False

    begin_time = time.time()

    if test_instance_id is None:
        print('Please set test instances.')
        return

    # 保存预测结果的路径
    save_filename = f'test_{test_instance_id}_checkpoint_{args.checkpoint_num}.npy'
    save_path = args.result_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file_path = os.path.join(save_path, save_filename)

    if os.path.exists(save_file_path):
        print(f'Return. Prediction of instance {test_instance_id} already exists.')
        return



    test_dataset = BrainDatasetTest_MSD(root_dir=test_data_dir,
                                        test_instance_id=test_instance_id,
                                        patch_size=args.patch_size,
                                        overlap_step=args.overlap_step,
                                        transform=transforms.Compose([
                                            RandomCrop_MSD(patch_size),
                                            TestToTensor_MSD()
                                        ]))

    # build test dataloader
    dataloaders_test = DataLoader(test_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.worker_num)


    # 设置模型为评估模式并禁用梯度计算
    model.eval()
    with torch.autograd.set_grad_enabled(False):
        for sample in dataloaders_test:
            # 直接加载整个图像
            input_image = sample['image'].to(device)
            input_image = input_image.float()
            # original_shape = sample['original_shape'].numpy().squeeze()  # 获取原始形状

            # 模型推理
            output = model(input_image)
            output_prob = torch.nn.functional.softmax(output, dim=1)  # 计算类别概率

            # 将预测结果移到CPU并转换为numpy数组
            output_prob = output_prob.cpu().numpy()
            # 获取每个体素的最终类别预测
            results = np.argmax(output_prob, axis=1).squeeze()

            # 保存预测结果
            np.save(save_file_path, results)
            print(f'{save_filename} saved.')

            # 计算时间
            time_elapsed = time.time() - begin_time
            print('Prediction complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

    print('Test DONE!')

