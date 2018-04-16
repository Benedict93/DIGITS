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


parser = argparse.ArgumentParser(description='Process model parameters in Pytorch')

# Basic Model Parameters
parser.add_argument('--batch_size', type=int, default=10,
                    help='input batch size for training (default: 10)')
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
                    help='SGD momentum (default: 0.9)')  # Not used by DIGITS front-end
parser.add_argument('--network', default='',
                    help='File containing network (model)')
parser.add_argument('--networkDirectory', default='',
                    help='Directory in which network exists')
parser.add_argument('--optimization', default='sgd',
                    choices=['sgd', 'nag', 'adagrad', 'rmsprop', 'adadelta', 'adam', 'sparseadam', 'adamax', 'asgd', 'rprop'],
                    help='Optimization method')
parser.add_argument('--loss', default='nll',
                    choices=['nll', 'mse', 'bse', 'pnll', 'cosemb', 'crossen', 'hingemeb', 'kldiv', 'l1', 'mr', 'mlm', 'mlsm','mm', 'bcelogits', 'sl1', 'sm', 'tm'],
                    help='Loss function')
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
parser.add_argument('--subtractMean', default='none', choices=['image', 'pixel', 'none'],
                    help="Select mean subtraction method. Possible values are 'image', 'pixel' or 'none'")
parser.add_argument('--train_db', default='',
                    help='Directory with training file source')
parser.add_argument('--train_labels', default='',
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
parser.add_argument('--lr_policy', default='fixed',
                    choices=['fixed', 'step', 'exp', 'inv', 'multistep', 'poly', 'sigmoid'],
                    help='Learning rate policy. (fixed, step, exp, inv, multistep, poly, sigmoid)')
parser.add_argument('--lr_gamma', type=float, default=-1,
                    help='Required to calculate learning rate. Applies to: (step, exp, inv, multistep, sigmoid)')
parser.add_argument('--lr_power', type=float, default=float('Inf'),
                    help='Required to calculate learning rate. Applies to: (inv, poly)')
parser.add_argument('--lr_stepvalues', default='',
                    help='Required to calculate stepsize of the learning rate. Applies to: (step, multistep, sigmoid). For the multistep lr_policy you can input multiple values seperated by commas')

# Augmentation: Other augmentations can be added in from torchvision.transforms package
parser.add_argument('--augFlip', default='none', choices=['none', 'fliplr', 'flipup', 'fliplrud'],
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

args = parser.parse_args()

args.cuda = torch.cuda.is_available()

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


def loadLabels(filename):
    with open(filename) as f:
        return f.readlines()

def train(epoch, model, train_loader, optimizer, criterion):
    losses = average_meter()
    accuracy = average_meter()
    initial_epoch = epoch
    epoch = float(epoch)
    log_interval = len(train_loader) / 10

    # Switch to train mode
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        # Compute output
        output = model(data)

        # Apply loss function and measure loss
        loss = F.nll_loss(output, target)
        losses.update(loss.data[0], data.size(0))

        # Measure accuracy
        pred = output.data.max(1)[1]
        prec = pred.eq(target.data).cpu().sum()
        accuracy.update(float(prec) / data.size(0), data.size(0))

        # Compute gradient and SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            if epoch.is_integer() == 0:
                print('Train Epoch: {}\t'
                     'Batch: [{:5d}/{:5d} ({:3.0f}%)]\t'
                     'Loss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 
                        100. * batch_idx / len(train_loader), losses.val))
                logging.info("Training (epoch " + str(epoch) + "):" + " loss = " + str(losses.val) + ", lr = " + str(args.lr_base_rate) + ", accuracy = {0:.2f}".format(accuracy.avg))
            epoch += 0.1

def validate(epoch, model, validation_loader, criterion):
    losses = average_meter()
    accuracy = average_meter()
    epoch = float(epoch) + 1

    # Switch to evaluate mode
    model.eval()

    for data, target in validation_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target, volatile=True)

        # Compute output
        output = model(data)

        # Apply loss function and measure loss
        loss = F.nll_loss(output, target)
        losses.update(loss.data[0], data.size(0))

        # Measure accuracy
        pred = output.data.max(1)[1]
        prec = pred.eq(target.data).cpu().sum()
        accuracy.update(float(prec) / data.size(0), data.size(0))

    print('\nTest: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        losses.avg, int(accuracy.sum), len(validation_loader.dataset), 100. * accuracy.avg))
    logging.info("Validation (epoch " + str(epoch) + "):" + " loss = " + str(losses.avg) + ", accuracy = " + "{0:.2f}".format(accuracy.avg))

class average_meter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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

    snapshot_prefix = args.snapshotPrefix if args.snapshotPrefix else args.network.split('.')[0]
    logging.info("Model weights will be saved as %s_<EPOCH>_Model.pth.tar", snapshot_prefix)

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

    transform = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])

    # Import the network file
    path_network = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.networkDirectory, args.network)
    exec(open(path_network).read(), globals())

    try:
        Net
    except NameError:
        logging.error("The user model class 'Net' is not defined.")
        exit(-1)
    if not inspect.isclass(Net):  # noqa
        logging.error("The user model class 'Net' is not a class.")
        exit(-1)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.train_db:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transform), batch_size=args.batch_size, shuffle=args.shuffle, **kwargs)
    if args.validation_db:
        validation_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, download=True,
                           transform=transform), batch_size=args.batch_size, shuffle=args.shuffle, **kwargs)
    model = Net()
    if args.cuda:
        model.cuda()

    # Optimizer - under torch.optim 
    if args.optimization == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr_base_rate, momentum=args.momentum)
    elif args.optimization =='nag':
        optimizer = optim.SGD(model.parameters(), lr=args.lr_base_rate,momentum=args.momentum, nesterov=True)
    elif args.optimization =='adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr_base_rate)
    elif args.optimization =='rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr_base_rate)
    elif args.optimization =='adadelta':
        optimizer = optim.adadelta(model.parameters(), lr=args.lr_base_rate)
    elif args.optimization =='adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr_base_rate)
    elif args.optimization =='sparseadam':
        optimizer = optim.SparseAdam(model.parameters(), lr=args.lr_base_rate)
    elif args.optimization =='adamax':
        optimizer = optim.Adamax(model.parameters(), lr=args.lr_base_rate)
    elif args.optimization =='asgd':
        optimizer = optim.ASGD(model.parameters(), lr=args.lr_base_rate)
    elif args.optimization =='rprop':
        optimizer = optim.Rprop(model.parameters(), lr=args.lr_base_rate)

    # Loss function - torch.nn 
    if args.loss == 'nll':
        criterion = nn.NLLLoss()
    elif args.loss =='mse':
        criterion = nn.MSELoss()
    elif args.loss =='bse':
        criterion = nn.BCELoss()
    elif args.loss =='pnll':
        criterion = nn.PoissonNLLLoss()
    elif args.loss =='cosemb':
        criterion = nn.CosineEmbeddingLoss()
    elif args.loss =='crossen':
        criterion = nn.CrossEntropyLoss()
    elif args.loss =='hingemeb':
        criterion = nn.HingeEmbeddingLoss()
    elif args.loss =='kldiv':
        criterion = nn.KLDivLoss()
    elif args.loss =='l1':
        criterion = nn.L1Loss()
    elif args.loss =='mr':
        criterion = nn.MarginRankingLoss()
    elif args.loss =='mlm':
        criterion = nn.MultiLabelMarginLoss()
    elif args.loss =='mlsm':
        criterion = nn.MultiLabelSoftMarginLoss()
    elif args.loss =='mm':
        criterion = nn.MultiMarginLoss()
    elif args.loss =='bcelogits':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss =='sl1':
        ocriterion = nn.SmoothL1Loss()
    elif args.loss =='sm':
        criterion = nn.SoftMarginLoss()
    elif args.loss =='tm':
        criterion = nn.TripletMarginLoss()


    
    logging.info('Started training the model')

    # Intiial forward pass
    validate(0, model, validation_loader, criterion)

    for epoch in range(0, args.epoch):

        # Training network
        train(epoch, model, train_loader, optimizer, criterion)

        # For every validation interval, perform validation
        if args.validation_db and epoch % args.validation_interval == 0:
            validate(epoch, model, validation_loader, criterion)

    # Final validation pass 
    validate(args.epoch, model, validation_loader, criterion)

if __name__ == '__main__':
        main()
