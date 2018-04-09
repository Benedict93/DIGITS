#! /usr/bin/env python2

""" 
Pytorch training executable for DIGITS

Defines the training procedure

Usage:
See the self-documenting flags below.

"""
from __future__ import print_function

import time

import datetime
import inspect
import json
import math
import numpy as np
import os

import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms


#define model parameters for the model in PyTorch?? Each to either have a initial value!
"""
#Basic Model Parameters

batch_size: Number of images to process in a batch
croplen: Crop (x and y). A zero value means no cropping will be applied
epoch: Number of epochs to train, -1 for unbounded
inference_db: Directory with inference file source
validation_interval: Number of train epochs to complete, to perform one validation
labels_list: Text file listing label definitions
momentum: value of 0.9, Momentum # Not used by DIGITS front-end
mean: Mean image file
network: File containing network (model)
networkDirectory: Directory in which network exists
optimization: Optimization method
save: Save directory
seed: Fixed input seed for repeatable experiments
shuffle: Shuffle records before training
snapshotInterval: Specifies the training epochs to be completed before taking a snapshot
SnapshotPrefix: Prefix of the weights/snapshots
subtractMean: Select mean subtraction method. Possible values are 'image', 'pixel' or 'none'
train_db: Directory with training file source
train_labels: Directory with an optional and seperate labels file source for training
validation_db: Directory with validation file source
validation_labels: Directory with an optional and seperate labels file source for validation
visualizeModelPath: Constructs the current model for visualization
visualize_inf: Will output weights and activations for an inference job.
weights: Filename for weights of a model to use for fine-tuning

#Augmentation
augFlip:The flip options {none, fliplr, flipud, fliplrud} as randompre-processing augmentation
augNoise:The stddev of Noise in AWGN as pre-processing augmentation
augContrast:The contrast factor's bounds as sampled from a random-uniform distribution
     as pre-processing  augmentation
augWhitenin:Performs per-image whitening by subtracting off its own mean and
    dividing by its own standard deviation.
augHSVhg:The stddev of HSV's Hue shift as pre-processing  augmentation
augHSVs:The stddev of HSV's Saturation shift as pre-processing  augmentation
augHSVv: The stddev of HSV's Value shift as pre-processing augmentation

"""
parser = argparse.ArgumentParser(description='Process model parameters in Pytorch')

# Basic Model Parameters
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training (default: 16)')
parser.add_argument('--croplen', type=int, default=0, 
                    help='Crop (x and y). A zero value means no cropping will be applied (default: 0)')
parser.add_argument('--epoch', type=int, default=10, 
                    help='Number of epochs to train, -1 for unbounded')
parser.add_argument('--inference_db', default='', 
                    help='Directory with inference file source')
parser.add_argument('--validation_interval', type=int, default=1, 
                    help='Number of train epochs to complete, to perform one validation (default: 1)')
parser.add_argument('--labels_list', default='', 
                    help='Text file listing label definitions')
parser.add_argument('--momentum', type=float, default=0.9, 
                    help='SGD momentum (default: 0.9)') # Not used by DIGITS front-end
parser.add_argument('--network', default='', 
                    help='File containing network (model)')
parser.add_argument('--networkDirectory', default='', 
                    help='Directory in which network exists')
parser.add_argument('--optimization', default='sgd', choices=['sgd','nag','adagrad','rmsprop','adadelta','adam','sparseadam','adamax','asgd','lbfgs','rprop'],
                    help='Optimization method')
parser.add_argument('--save', default='results', 
                    help='Save directory')
parser.add_argument('--seed', type=int, default=0, 
                    help='Fixed input seed for repeatable experiments (default:0)')
parser.add_argument('--shuffle', action='store_true', default=False, 
                    help='Shuffle records before training')
parser.add_argument('--snapshotInterval', type=float, default=1.0, 
                    help='Specifies the training epochs to be completed before taking a snapshot')
parser.add_argument('--snapshotPrefix', default='',
                    help='Prefix of the weights/snapshots')
parser.add_argument('--subtractMean', default='none', choices=['image','pixel','none'],
                    help="Select mean subtraction method. Possible values are 'image', 'pixel' or 'none'")
parser.add_argument('--train_db', default='',
                    help='Directory with training file source')
parser.add_argument('--train_labels',  default='',
                    help='Directory with an optional and seperate labels file source for training')
parser.add_argument('--validation_db', default='', 
                    help='Directory with validation file source')
parser.add_argument('--validation_labels', default='',
                    help='Directory with an optional and seperate labels file source for validation')
parser.add_argument('--visualizeModelPath', default='',
                    help='Constructs the current model for visualization')
parser.add_argument('--visualize_inf', action='store_true', default=False, 
                    help='Will output weights and activations for an inference job.')
parser.add_argument('--weights', default='',
                    help='Filename for weights of a model to use for fine-tuning')


parser.add_argument('--bitdepth', type=int, default=8,
                    help='Specifies an image bitdepth')

parser.add_argument('--lr_base_rate', type=float, default=0.01,
                    help='Learning Rate')
parser.add_argument('--lr_policy', default='fixed', choices=['fixed','step','exp','inv','multistep','poly','sigmoid'], 
                    help='Learning rate policy. (fixed, step, exp, inv, multistep, poly, sigmoid)')
parser.add_argument('--lr_gamma', type=float, default=-1,
                    help='Required to calculate learning rate. Applies to: (step, exp, inv, multistep, sigmoid)')
parser.add_argument('--lr_power', type=float, default=float('Inf'),
                    help='Required to calculate learning rate. Applies to: (inv, poly)')
parser.add_argument('--lr_stepvalues', default='',
                    help='Required to calculate stepsize of the learning rate. Applies to: (step, multistep, sigmoid). For the multistep lr_policy you can input multiple values seperated by commas')


#Augmentation
parser.add_argument('--augFlip', default='none', choices =['none', 'fliplr', 'flipup', 'fliplrud'],
                    help='The flip options {none, fliplr, flipud, fliplrud} as randompre-processing augmentation')
parser.add_argument('--augNoise', type=float, default=0,
                    help='The stddev of Noise in AWGN as pre-processing augmentation')
parser.add_argument('--augContrast', type=float, default=0,
                    help='The contrast factors bounds as sampled from a random-uniform distribution as pre-processing  augmentation')
parser.add_argument('--augWhitening', action='store_true', default=False, 
                    help='Performs per-image whitening by subtracting off its own mean and dividing by its own standard deviation')
parser.add_argument('--augHSVhg', type=float, default=0,
                    help='The stddev of HSV Hue shift as pre-processing  augmentation')
parser.add_argument('--augHSVs', type=float, default=0,
                    help='The stddev of HSV Saturation shift as pre-processing  augmentation')
parser.add_argument('--augHSVv', type=float, default=0,
                    help='The stddev of HSV Value shift as pre-processing augmentation')

""" 
Other augmentations to be added in from torchvision.transforms package

"""
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO) 
                    
def loadLabels(filename):
    with open(filename) as f:
        return f.readlines()

args = parser.parse_args()

args.cuda = torch.cuda.is_available()

def main():
    if args.validation_interval == 0:
        args.validation_db = None
    if args.seed:
        torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    batch_size_train = args.batch_size  
    batch_size_val = args.batch_size
    logging.info("Train batch size is %s and validation batch size is %s", batch_size_train, batch_size_val)

    # This variable keeps track of next epoch, when to perform validation.
    next_validation = args.validation_interval
    logging.info("Training epochs to be completed for each validation : %s", next_validation)

    # This variable keeps track of next epoch, when to save model weights.
    next_snapshot_save = args.snapshotInterval
    logging.info("Training epochs to be completed before taking a snapshot : %s", next_snapshot_save)
    last_snapshot_save_epoch = 0 

    snapshot_prefix = args.snapshotPrefix if args.snapshotPrefix else args.network.split('.')[0]
    logging.info("Model weights will be saved as %s_<EPOCH>_Model.pt", snapshot_prefix)

    if not os.path.exists(args.save):
        os.makedirs(args.save)
        logging.info("Created a directory %s to save all the snapshots", args.save)

    classes = 0
    nclasses = 0
    if args.labels_list:
        logging.info("Loading label definitions from %s file", args.labels_list)
        classes = loadLabels(args.labels_list)
        nclasses = len(classes)
        if not classes:
            logging.error("Reading labels file %s failed.", args.labels_list)
            exit(-1)
        logging.info("Found %s classes", nclasses)

    # Import the network file
    path_network = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.networkDirectory, args.network)
    exec(open(path_network).read(), globals())

    try:
        LeNet
    except NameError: 
        logging.error("The user model class 'LeNet' is not defined.")
        exit(-1)
    if not inspect.isclass(LeNet):  # noqa
        logging.error("The user model class 'LeNet' is not a class.")
        exit(-1)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.train_db:
        train_loader = torch.utils.data.DataLoader(args.train_db, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs)
    if args.validation_db:
        validation_loader = torch.utils.data.DataLoader(
            args.validation_db, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs)

    model = LeNet()
    if args.cuda:
        model.cuda()

    if args.optimization == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr_base_rate, momentum=args.momentum)

    for epoch in range(1, args.epoch + 1):
        train(epoch, model, train_loader, optimizer)


def train(epoch, model, train_loader, optimizer):
    model.train()
    log_interval = 10
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.data[0]))
"""
def test(model, validation_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(validation_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(validation_loader.dataset),
        100. * correct / len(validation_loader.dataset)))
"""


if __name__ == '__main__':
    main()
