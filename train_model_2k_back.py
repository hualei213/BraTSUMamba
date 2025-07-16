import time
import torch.autograd
import torch.cuda
import torch.nn as nn
import torch.nn.functional
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import copy
import os
import math
import torch

from dataset_builder import BrainDataset, BrainDatasetTest, RandomCrop, TestToTensor, ToTensor
from config import get_args

import os

from utils.logger import generate_logger, get_logger

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
train_dataset = BrainDataset(root_dir=train_data_dir,
                             transform=transforms.Compose([
                                 RandomCrop(patch_size),
                                 ToTensor()
                             ]))
# build validation dataset
val_dataset = BrainDataset(root_dir=val_data_dir,
                           transform=transforms.Compose([
                               RandomCrop(patch_size),
                               ToTensor()
                           ]))

dataloaders_train = DataLoader(train_dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=num_works,
                               pin_memory=True)
dataloaders_val = DataLoader(val_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_works,
                             pin_memory=True)

dataloaders = {'train': dataloaders_train,
               'val': dataloaders_val}
# get size of dataset
dataset_sizes = {'train': len(train_dataset),
                 'val': len(val_dataset)}

print('train dataset size: {}, '
      'val dataset size: {}'.format(dataset_sizes['train'],
                                    dataset_sizes['val']))


####################################################################################
def train_model(model,
                criterion,
                optimizer,
                scheduler,
                num_epochs,
                resume_train=True,
                epochs_per_val=5000,
                batch_size=1,
                lr=1e-3,
                result_dir='./result',
                save_cp=True,
                device=torch.device('cpu')):
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
    train_log_writer = generate_logger(str(result_dir).replace(".","").replace("/","")+"_"
                                       +"train_epoch-" + str(num_epochs) + "_" +
                                       "bs-" + str(batch_size) + "_"+
                                       "lr-" + str(lr)
                                       )
    # create val logger
    val_log_writer = generate_logger(str(result_dir).replace(".","").replace("/","")+"_"+
                                     "val_epoch-" + str(num_epochs) + "_" +
                                     "bs-" + str(batch_size) + "_" +
                                     "lr-" + str(lr)
                                     )
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
        Learning rate: {}
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

        # Iterate over test_data.
        for itr, sample in enumerate(dataloaders[phase]):
            batch_start = time.time()
            input_images = sample['image'].to(device)
            labels = sample['label'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.autograd.set_grad_enabled(phase == 'train'):
                # output from the model
                output_images = model(input_images)
                # print("++++++++++++++++++++++++++++++++++")
                # print("output_images--->"+str(output_images.shape))
                # predicted classes
                preds_classes = torch.argmax(output_images, dim=1)
                # print("preds_classes--->"+str(preds_classes.shape))
                # predicted probabilities
                preds_prob = torch.nn.functional.softmax(output_images, dim=1)
                # print("preds_prob--->"+str(preds_prob.shape))
                # expand dimension
                preds_classes = torch.unsqueeze(preds_classes, dim=1)
                # print("preds_classes--->"+str(preds_classes.shape))
                #### labels = torch.nn.functional.softmax(labels)
                # print(labels.shape)
                preds_target = torch.squeeze(labels.long(), dim=1)
                # print("preds_target---->"+str(preds_target.shape))

                loss = criterion(preds_prob, preds_target)

                # backward + optimize only if in training phase
                if phase == 'train':
                    # calculate gradients for this batch
                    loss.backward()
                    # update model
                    optimizer.step()
                    # update scheduler
                    # scheduler.step()

            # statistics
            batch_loss += loss.item() * input_images.size(0)
            pred_arr = preds_classes.byte() == labels.byte()
            batch_corrects += torch.mean(pred_arr.float()).item() * input_images.size(0)
            batch_acc = torch.mean(pred_arr.float()).item()

            batch_end = time.time()
            print('Epoch: [{}/{}]'
                  '\tIteration: [{}/{}]'
                  '\tTime {:.3f}'
                  '\tLoss {:.4f}'
                  '\tAcc {:.4f}'
                  .format(epoch, num_epochs - 1,
                          itr, math.ceil(dataset_sizes[phase] / batch_size),
                          batch_end - batch_start,
                          loss.item(),
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

        # print('{} average loss: {: .4f}, average acc: {: .4f}'.format(
        #     phase, epoch_loss, epoch_acc
        # ))

        # display time
        epoch_time = time.time() - epoch_start
        total_time = time.time() - since
        print('Epoch time: {:.0f}m {:.4f}s, '
              'Total time: {:.0f}m {:.0f}s'.format(epoch_time // 60,
                                                   epoch_time % 60,
                                                   total_time // 60,
                                                   total_time % 60
                                                   ))
        #####################################################################
        # continue to train
        # it isn't time for evaluation
        # or save the checkpoint
        if epoch % epochs_per_val != 0:
            continue
        else:
            # save checkpoint
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                        }, checkpoint_path)
            print('Save checkpoint at epoch %d' % epoch)

        #####################################################################
        # validation once
        phase = 'val'
        model.eval()
        batch_loss = 0.0
        batch_corrects = 0.0

        # Iterate over test_data.
        for sample in dataloaders[phase]:
            input_images = sample['image'].to(device)
            labels = sample['label'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.autograd.set_grad_enabled(phase == 'train'):
                # output from the model
                output_images = model(input_images)
                # predicted classes
                preds_classes = torch.argmax(output_images, dim=1)
                # predicted probabilities
                preds_prob = torch.nn.functional.softmax(output_images, dim=1)
                # expand dimension
                preds_classes = torch.unsqueeze(preds_classes, dim=1)
                loss = criterion(preds_prob, torch.squeeze(labels.long(), dim=1))

            # statistics
            batch_loss += loss.item() * input_images.size(0)
            pred_arr = preds_classes.byte() == labels.byte()
            batch_corrects += torch.mean(pred_arr.float()).item() * input_images.size(0)

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
        if epoch % epochs_per_val == 0:
            model_path = os.path.join(result_dir,
                                      'model-%d' % epoch)
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, model_path)
            print('Save model-%d.' % epoch)

            if epoch_acc > best_acc:
                early_stop = 0
                best_acc = epoch_acc
                model_path = os.path.join(result_dir,
                                          'best_model-%d' % epoch)
                # best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, model_path)
                print('Save best acc model at epoch %d.' % epoch)

            early_stop += 1
            # if early_stop > num_epochs / epochs_per_val / 5:
            #     print('model stop at %d.' % epoch)
            #     break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    ## Question: which model is best?
    # load the best model weights
    model.load_state_dict(best_model_wts)
    return model
