
"""
Digits default PyTorch Ops as helper functions.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

STAGE_TRAIN = 'train'
STAGE_VAL = 'val'


def mse_loss(lhs, rhs):
    loss = nn.MSELoss().(Lhs,rhs)
    return loss
    
def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])


def hwc_to_chw(x):
    return tf.transpose(x, [2, 0, 1])


def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])


def chw_to_hwc(x):
    return tf.transpose(x, [1, 2, 0])


def bgr_to_rgb(x):
    return tf.reverse(x, [2])


def rgb_to_bgr(x):
    return tf.reverse(x, [2])


def get_available_gpus():
    """
    Queries the CUDA GPU devices visible to Tensorflow.
    Returns:
        A list with tf-style gpu strings (f.e. ['/gpu:0', '/gpu:1'])
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']