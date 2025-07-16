import datetime
import os.path
import random
import shutil

import torch
from tensorboardX import SummaryWriter

def generate_logger(log_path,mode):
    time = datetime.datetime.now()
    nowtime = time.strftime("%m-%d-%H-%M")
    out_log_dir= os.path.join(log_path,"logs","{}_{}".format(nowtime,mode))
    os.makedirs(out_log_dir,exist_ok=True)
    writer = get_logger(out_log_dir)
    return writer

def get_logger(out_log_dir):
    writer = SummaryWriter(out_log_dir)
    return writer


# #############################################################
# writer,path = generate_logger("test")
# for i in range(100):
#     writer.add_scalar('example1', i ** 2, global_step=i)
# checkpoint_path = os.path.join("./", 'checkpoint')
# torch.save(
#     {
#         "path":path
#     },checkpoint_path
# )
# ar=torch.load(checkpoint_path)
# path= ar["path"]
# print(path)
# writer = get_logger(path)
# for i in range(100,200):
#     writer.add_scalar('example1', i, global_step=i)
#
#
