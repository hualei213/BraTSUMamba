import os.path

import torch
from tensorboardX import SummaryWriter

import unet
from unet import *

'''
    create model structure graph
    :param model_name: input model name
    :param input_data_to_model: test_data type is a tensor
'''


def get_model_graph(model_name="temp", model=None, input_data_to_model=None):
    root_path = "./models/"
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if input_data_to_model is None or model is None:
        return "No test_data input or model"
    # create model path
    log_dir = root_path + model_name
    graph_writer = SummaryWriter(log_dir=log_dir)
    # get model graph
    print("----Start create model {}----".format(model_name))
    graph_writer.add_graph(model=model, input_to_model=input_data_to_model)
    # colse SummaryWriter method
    graph_writer.close()
    print("Model {} Structure Graph Done!".format(model_name))


if __name__ == "__main__":
    from unet.UNeXt_3D import UNeXt3D
    model = UNeXt3D(1, 2)
    input_data = torch.randn((1, 1, 128, 128, 128))

    get_model_graph("UNeXt3D", model=model, input_data_to_model=input_data)