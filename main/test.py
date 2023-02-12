import easyfl
import time
from torch import nn

from easyfl.client.base import BaseClient
import  torch
import torch.nn.functional as F
import copy
from torch.autograd import Variable
from utils import accuracy
from local_cnn import CNNModel
from local_resnet import ResModel

if __name__=='__main__':
    
    para_state_dict = torch.load("./saved_models/_global_model_r_999.pth")
    model=CNNModel()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in para_state_dict.items() if k in model_dict and "fc"  in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)