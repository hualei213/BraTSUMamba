import csv
import time

import h5py
import numpy as np
import torch.autograd
import torch.cuda
import torch.nn as nn
import torch.nn.functional
from monai.transforms import MapLabelValued
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import copy
import os
import math
import torch
import itertools
import torch.nn.functional as F
from dataset_builder import BrainDataset, BrainDatasetTest, RandomCrop, ToTensor,BrainDataset_MSD,BrainDatasetVal_MSD,RandomCrop_MSD,ToTensor_MSD,ValToTensor_MSD
# from dataset_builder.BrainDataset import BrainDataset_nii,RandomCrop_nii,ToTensor_nii
from config import get_args

import os

from Criterion import BinaryDiceLoss, BinaryDice

from dataset_builder.BrainDatasetVal import BrainDatasetVal, ValToTensor
from utils.logger import generate_logger, get_logger
from utils.test import get_percentile_distance, get_ASSD

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import pdb

#####################################################################################
# set parameters

args = get_args()
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

#####################################################################################
# prepare dataset
# build train dataset

train_dataset = BrainDataset_MSD(root_dir=train_data_dir,
                             transform=transforms.Compose([
                                 RandomCrop_MSD(patch_size),
                                 ToTensor_MSD()
                             ]))
val_dataset = BrainDatasetVal_MSD(root_dir=val_data_dir,
                              test_instance_id=test_instance_id,
                              patch_size=args.patch_size,
                              overlap_step=args.overlap_step,
                              transform=transforms.Compose([
                                  ValToTensor_MSD()
                              ]))



dataloaders_train = DataLoader(train_dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=num_works,
                               pin_memory=True)
dataloaders_val = DataLoader(val_dataset,
                             batch_size=1,
                             shuffle=True,
                             num_workers=num_works,
                             pin_memory=True)



dataloaders = {'train': dataloaders_train,
               'val': dataloaders_val}
# get size of dataset
dataset_sizes = {'train': len(train_dataset),
                 'val': len(val_dataset)}

print('train dataset size: {},'
      'val dataset size: {}'.format(dataset_sizes['train'],
                                    dataset_sizes['val']))


####################################################################################
bin_acc = BinaryDice()

def process_label(label):
    ncr = label == 1
    ed = label == 2
    et = label == 3     #BraTS2023
    # et =label == 4        #BraTS2020
    ET = et
    TC = ncr + et
    WT = ncr + et + ed
    return ET, TC, WT


def val_sample(sample, model, device):
    batch_size = 1
    # predicted probability of all patches
    preds = []
    # index of all patches
    patch_ids = []

    input_images = sample['image'].to(device)

    patch_id = sample['patch_id']

    patch_id = patch_id.type(torch.int16)
    patch_id = torch.squeeze(patch_id, dim=0)

    original_shape = sample['original_shape'].numpy()
    original_shape = np.squeeze(original_shape)


    batch_group_mile_stone = list(range(0, patch_id.size(0), batch_size))
    if patch_id.size(0) % batch_size != 0:
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



            input_patches_id = [patch_id[j, :].numpy()
                                for j in list(range(batch_group_mile_stone[i],
                                                    batch_group_mile_stone[i + 1]))]

            input_patches = torch.stack(input_patches)
            input_patches = torch.squeeze(input_patches, 1)


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
    patch_ids = torch.tensor(np.array(patch_ids))
    # move to CPU
    preds = preds.to('cpu')
    patch_ids = patch_ids.to('cpu')

    # processing patch by patch
    # begin_time = time.time()
    # print('-' * 20 + 'Processing: ' + str(test_instance_id) + '-' * 20)
    # all_predictions: accumulated prediction probabilities of the whole image
    # shape: class_number x image_width x image_height x image_depth
    # numpy image: H x W x C
    # torch image: C X H X W
    all_predictions = torch.zeros(preds.size(1), original_shape[0], original_shape[1], original_shape[2]).to('cpu')

    patch_num = patch_ids.size(0)
    for i in range(patch_num):
        # print('Processing patch %d/%d' % (i, patch_num),
        #       end='\r',
        #       flush=True)
        p3d = (
            patch_ids[i][2].item(), original_shape[2] - patch_ids[i][2].item() - patch_size, patch_ids[i][1].item(),
            original_shape[1] - patch_ids[i][1].item() - patch_size, patch_ids[i][0].item(),
            original_shape[0] - patch_ids[i][0].item() - patch_size)
        # expand size of preds to class_number x image_width x image_height x image_depth, padded with 0
        pad_patch_tensor = F.pad(preds[i, :], p3d, "constant", 0)
        # add to all_predictions
        all_predictions = all_predictions + pad_patch_tensor

    values, results = torch.max(all_predictions, 0)
    # convert tensor to numpy array
    results = results.data.numpy()

    return torch.unsqueeze(all_predictions, dim=0)


def train_model(model,
                criterion,
                optimizer,
                scheduler,
                num_epochs,
                resume_train=True,
                epochs_per_val=50,
                batch_size=1,
                lr=1e-3,
                result_dir='./result',
                save_cp=True,
                device=torch.device('cpu'),
                log_dir="./"):
    """
        train the model
        :param model: model to be trained
        :param criterion: the objective function
        :param optimizer: optimization method
        :param scheduler:
        :param num_epochs: number of training epochs
               resume_train: whether to train from save checkpoint, default is Falses
        :param epochs_per_val: number of training epochs to run before one evolution
        :param batch_size: number of samples contained in one batch
        :param lr: learning rate
        :param result_dir: directory to save the trained model
        :param save_cp:
        :return: the trained model
        """

    # torch.backends.cudnn.enable = True
    # torch.backends.cudnn.benchmark = True
    # create train logger

    # create train logger
    # train_log_writer = generate_logger(str(result_dir).replace(".","").replace("/","")+"_"
    #                                    +"train_epoch-" + str(num_epochs) + "_" +
    #                                    "bs-" + str(batch_size) + "_"+
    #                                    "lr-" + str(lr)
    #                                    )
    train_log_writer = generate_logger(log_dir, "train")
    # create val logger
    val_log_writer = generate_logger(log_dir, "val")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    checkpoint_path = os.path.join(result_dir, 'checkpoint')
    since = time.time()
    # pdb.set_trace()
    #############################################################################
    # load model and parameters from checkpoint
    if resume_train:
        if not os.path.exists(checkpoint_path):
            begin_epoch = 0
            print('No checkpoint available. Start training from epoch 0.')
        else:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            begin_epoch = checkpoint['epoch']
            loss = checkpoint['loss']
    else:
        begin_epoch = 0

    # deep copy model parameter to get current best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print('''
    Starting training:
        From Epoch: {}
        Remaining Epochs: {}
        Total Epochs: {}
        Patch size: {}
        Batch size: {}
        Learning rate_seg: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
    '''.format(begin_epoch,
               num_epochs - begin_epoch,
               num_epochs,
               patch_size,
               batch_size,
               lr,
               len(train_dataset),
               len(val_dataset),
               str(save_cp)))
    print('#' * 50)
    early_stop = 0
    # train / validation epoch
    for epoch in range(begin_epoch, num_epochs):
        epoch_start = time.time()
        print('-' * 100)
        # print('Epoch {}/{}.'.format(epoch, num_epochs - 1))

        #####################################################################
        # training cycle
        # print('Start a training cycle')
        phase = 'train'
        # scheduler.step()
        model.train()
        batch_loss = 0.0
        batch_corrects = 0.0
        dic = {}

        # Iterate over test_data.
        for itr, sample in enumerate(dataloaders[phase]):
            batch_start = time.time()
            input_images = sample['image'].to(device)
            labels = sample['label'].to(device)


            # idx = sample['id']
            # zero the parameter gradients
            optimizer.zero_grad()


            # forward
            # track history if only in train
            with torch.autograd.set_grad_enabled(phase == 'train'):
                # output from the model
                output_images = model(input_images)


                # predicted classes
                preds_classes = torch.argmax(output_images, dim=1)
                label_pred = torch.unsqueeze(preds_classes,dim=1)
                # print("preds_classes--->"+str(preds_classes.shape))
                # predicted probabilities
                preds_prob = torch.nn.functional.softmax(output_images, dim=1)
                # expand dimension
                preds_classes = torch.unsqueeze(preds_classes, dim=1)
                # print(labels.shape)
                preds_target = torch.squeeze(labels.long(), dim=1)
                loss_seg = criterion(preds_prob, preds_target)

                # 计算wt的acc
                label_et, label_tc, label_wt = process_label(labels)
                infer_et, infer_tc, infer_wt = process_label(preds_classes)

                # acc = BinaryDiceLoss(infer_wt,label_wt)
                acc = bin_acc(infer_wt, label_wt)

                batch_acc = acc.item()
                batch_corrects += acc.item() * input_images.size(0)

                # backward + optimize only if in training phase
                if phase == 'train':
                    # calculate gradients for this batch
                    loss = loss_seg
                    loss.backward()
                    # update model
                    optimizer.step()

            # statistics
            batch_loss += loss.item() * input_images.size(0)

            batch_end = time.time()

            print('Epoch: [{}/{}]'
                  '\tIteration: [{}/{}]'
                  '\tTime {:.3f}'
                  '\tLoss {:.4f}'
                  '\tLoss_seg {:.4f}'
                  '\tAcc {:.4f}'
                  .format(epoch, num_epochs - 1,
                          itr, math.ceil(dataset_sizes[phase] / batch_size),
                          batch_end - batch_start,
                          loss.item(),
                          loss_seg.item(),
                          batch_acc
                          )
                  )


        epoch_loss = batch_loss / dataset_sizes[phase]
        epoch_acc = batch_corrects / dataset_sizes[phase]
        # add epoch_loss to log
        train_log_writer.add_scalar("Loss/train", float(epoch_loss), global_step=int(epoch))
        train_log_writer.flush()
        # add epoch_acc to log
        train_log_writer.add_scalar("Accuracy/train", float(epoch_acc), global_step=int(epoch))
        train_log_writer.flush()


        # display time
        epoch_time = time.time() - epoch_start
        total_time = time.time() - since
        print('Epoch time: {:.0f}m {:.4f}s,'
              'Total time: {:.0f}m {:.0f}s'.format(epoch_time // 60,
                                                   epoch_time % 60,
                                                   total_time // 60,
                                                   total_time % 60
                                                   ))

        #####################################################################
        # continue to train
        # it isn't time for evaluation
        # or save the checkpoint
        if (epoch+1) % epochs_per_val == 0 or epoch==(num_epochs-1):
            # save checkpoint
            torch.save({'epoch': (epoch+1),
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                        }, checkpoint_path)

            print('Saved checkpoint at epoch %d' % (epoch+1))

        else:
            continue

        # # save checkpoint
        # torch.save({'epoch': (epoch + 1),
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'loss': loss
        #             }, checkpoint_path)
        #
        # print('Saved checkpoint at epoch %d' % (epoch + 1))

        # model_path = os.path.join(result_dir,
        #                           'model-%d' % (epoch + 1))
        # best_model_wts = copy.deepcopy(model.state_dict())
        # torch.save(best_model_wts, model_path)
        # print('Save model-%d.' % (epoch + 1))

        #####################################################################
        # validation once
        phase = 'val'
        model.eval()
        batch_loss = 0.0
        batch_corrects = 0.0
        ###############################新改部分
        # zero the parameter gradients
        optimizer.zero_grad()
        with torch.autograd.set_grad_enabled(False):
            for sample in dataloaders[phase]:
                ## 将数据送入，返回输出
                output_images_seg = val_sample(sample, model=model, device=device)
                preds_classes_seg = torch.argmax(output_images_seg, dim=1)
                # preds_classes_reg = torch.argmax(output_images_reg, dim=1)
                # predicted probabilities
                preds_prob = torch.nn.functional.softmax(output_images_seg, dim=1)
                # expand dimension
                preds_classes_seg = torch.unsqueeze(preds_classes_seg, dim=1)
                labels = sample["label"]
                loss = criterion(preds_prob, torch.squeeze(labels.long(), dim=1))
                # statistics
                batch_loss += loss.item() * sample["image"].size(0)
                pred_arr = preds_classes_seg.byte() == labels.byte()
                batch_corrects += torch.mean(pred_arr.float()).item() * sample["image"].size(0)

        epoch_loss = batch_loss / dataset_sizes[phase]
        epoch_acc = batch_corrects / dataset_sizes[phase]
        # add val_loss to log
        val_log_writer.add_scalar("Loss/validation", epoch_loss, global_step=epoch)
        val_log_writer.flush()
        # add val_acc to log
        val_log_writer.add_scalar("Accuracy/validation", epoch_acc, global_step=epoch)
        val_log_writer.flush()
        #####################################################################

        # save model, every <epochs_per_val> epochs
        # need save model too much so set name == num_epochs
        print("epoch-acc={}".format(epoch_acc))
        dic["epoch"] = epoch+1
        dic["acc"] = epoch_acc
        csv_path_acc = args.csv_dir
        with open(csv_path_acc, 'a', newline='') as csvfile:
            # 定义字段名
            fieldnames = ['epoch', 'acc']
            # 创建DictWriter实例
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # 如果文件是新的，则写入表头
            if epoch == 0:
                writer.writeheader()
            # 写入字典数据
            writer.writerow(dic)

        if (epoch+1) % epochs_per_val == 0 or epoch==(num_epochs-1):
            model_path = os.path.join(result_dir,
                                      'model-%d' % (epoch+1))
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, model_path)
            print('Save model-%d.' % (epoch+1))

            if epoch_acc > best_acc:
                early_stop = 0
                best_acc = epoch_acc
                model_path = os.path.join(result_dir,
                                          'best_model-%d' % (epoch+1))
                # best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, model_path)
                print('Save best acc model at epoch: %d Best acc: %f.' % ((epoch+1), best_acc))

            early_stop += 1


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    ## Question: which model is best?
    # load the best model weights
    model.load_state_dict(best_model_wts)
    return model
