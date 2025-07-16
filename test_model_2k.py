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
    make prediction with the model
    Now the model can make prediction for multiple test instances.
    But each instance is predicted with batchsize 1
    :param model: the trained model
    :return: list of file names
    """

    torch.backends.cudnn.benchmark = False

    begin_time = time.time()

    if test_instance_id == None:
        print('Please set test instances.')
        return

    save_filename = 'test_%s_checkpoint_%d.npy' % (test_instance_id, args.checkpoint_num)

    #######################################################################
    save_path = args.result_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if args.pred_sample:
        save_path = os.path.join(args.result_dir,"pred_files")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    ########################################################################
    save_file_path = os.path.join(save_path, save_filename)

    if os.path.exists(save_file_path):
        print('Return. Prediction of instance %s already exists.' % test_instance_id)
        return
    #####################################################################
    # build test dataset
    # the dataset contains only one test instance
    # atlas_dir=args.atlas,
    # test_dataset = BrainDatasetTest(root_dir=test_data_dir,
    #                                 test_instance_id=test_instance_id,
    #                                 patch_size=args.patch_size,
    #                                 overlap_step=args.overlap_step,
    #                                 transform=transforms.Compose([
    #                                     TestToTensor()
    #                                 ]))
    test_dataset = BrainDatasetTest_MSD(root_dir=test_data_dir,
                                    test_instance_id=test_instance_id,
                                    patch_size=args.patch_size,
                                    overlap_step=args.overlap_step,
                                    transform=transforms.Compose([
                                        TestToTensor_MSD()
                                    ]))

    # build test dataloader
    dataloaders_test = DataLoader(test_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.worker_num)

    # Iterate over test_data.
    with torch.autograd.set_grad_enabled(False):
        # predict instance by instance
        # dataloader returns 'sample'
        for sample in dataloaders_test:
            # predicted probability of all patches
            preds = []
            # index of all patches
            patch_ids = []

            # get prediction id
            print('-' * 20 + 'Load ' + str(test_instance_id) + '-' * 20)
            # get input patches
            # image_t1c = sample['image_t1c'].to(device)
            # image_t1n = sample['image_t1n'].to(device)
            # image_t2f = sample['image_t2f'].to(device)
            # image_t2w = sample['image_t2w'].to(device)
            input_images = sample['image'].to(device)




            # input_images = torch.cat([image_t1c, image_t1n, image_t2f, image_t2w], dim=1).to(device)


            # get ids of input patches
            patch_id = sample['patch_id']
            patch_id = patch_id.type(torch.int16)
            patch_id = torch.squeeze(patch_id, dim=0)

            original_shape = sample['original_shape'].numpy()
            original_shape = np.squeeze(original_shape)
            # cut_size = sample['cut_size'].numpy()
            # cut_size = np.squeeze(cut_size)

            #####################################################################
            # get predictions of all patches
            # separate batches into groups
            # each group contains #batch_size patches
            batch_group_mile_stone = list(range(0, patch_id.size(0), batch_size))
            if patch_id.size(0) % batch_size != 0:
                # add id of the final patch
                batch_group_mile_stone.append(patch_id.size(0))

            for i in range(len(batch_group_mile_stone)):
                # skip the final patch id
                if i == len(batch_group_mile_stone) - 1:
                    pass
                else:
                    print('Evaluate %dth batch, total %d'
                          % (i + 1, len(batch_group_mile_stone)))
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
                    output_patches = model(input_patches)
                    # predicted probabilities
                    output_patches_prob = torch.nn.functional.softmax(output_patches, dim=1)
                    # move tensor to cpu
                    output_patches_prob = output_patches_prob.to('cpu')
                    # convert to numpy array
                    preds.append(output_patches_prob)
                    patch_ids.append(input_patches_id)
                    # release GPU memory
                    torch.cuda.empty_cache()
                    del input_patches, \
                        output_patches_prob, \
                        output_patches

            # statistics
            predictions = {}  # create an empty dictionary

            # flatten lists
            preds = list(itertools.chain(*preds))
            patch_ids = list(itertools.chain(*patch_ids))
            # convert to torch.tensor
            preds = torch.stack(preds)
            patch_ids = torch.tensor(patch_ids)
            # move to CPU
            preds = preds.to('cpu')
            patch_ids = patch_ids.to('cpu')

            # processing patch by patch
            # begin_time = time.time()
            print('-' * 20 + 'Processing: ' + str(test_instance_id) + '-' * 20)
            # all_predictions: accumulated prediction probabilities of the whole image
            # shape: class_number x image_width x image_height x image_depth
            # numpy image: H x W x C
            # torch image: C X H X W
            all_predictions = torch.zeros(preds.size(1), original_shape[0], original_shape[1], original_shape[2]).to('cpu')

            patch_num = patch_ids.size(0)
            for i in range(patch_num):
                print('Processing patch %d/%d' % (i, patch_num),
                      end='\r',
                      flush=True)
                p3d = (
                patch_ids[i][2].item(), original_shape[2] - patch_ids[i][2].item() - patch_size, patch_ids[i][1].item(),
                original_shape[1] - patch_ids[i][1].item() - patch_size, patch_ids[i][0].item(),
                original_shape[0] - patch_ids[i][0].item() - patch_size)
                # expand size of preds to class_number x image_width x image_height x image_depth, padded with 0
                pad_patch_tensor = F.pad(preds[i, :], p3d, "constant", 0)
                # add to all_predictions
                all_predictions = all_predictions + pad_patch_tensor

            # convert probability to class results
            print('Getting classes results...')
            values, results = torch.max(all_predictions, 0)
            # convert tensor to numpy array
            results = results.data.numpy()

            # saving results
            print('Saving results...')
            np.save(save_file_path, results)
            print(save_filename + ' saved.')

            time_elapsed = time.time() - begin_time
            print('Prediction complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

    print('Test DONE!')
