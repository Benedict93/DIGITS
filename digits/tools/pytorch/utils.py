
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


