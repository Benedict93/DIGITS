# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
#
# This document should comply with PEP-8 Style Guide
# Linter: pylint

"""
Interface for setting up and creating a model in Tensorflow.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

# Local imports
import utils as digits

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


class Model(object):
    """
    Wrapper around the actual tensorflow workflow process.
    This is structured in a way that the user should only care about
    creating the model while using the DIGITS UI to select the
    optimizer and other options.

    This class is executed to start a tensorflow workflow.
    """
    def __init__(self, model, croplen, nclasses, optimization=None, momentum=None):
        self.model = model
        self.croplen = croplen
        self.nclasses = nclasses
        self.dataloader = None

        self._optimization = optimization
        self._momentum = momentum
        self._train = None

        # Touch to initialize
        # if optimization:
        #     self.learning_rate
        #     self.global_step
        #     self.optimizer

    def create_dataloader(self, db_path):
        self.dataloader = pt_data.LoaderFactory.set_source(db_path)
        self.dataloader.stage = self.stage
        self.dataloader.croplen = self.croplen
        self.dataloader.nclasses = self.nclasses

    def train(self):
        return self._train

    def learning_rate(self):
        # @TODO(tzaman): the learning rate is a function of the global step, so we could
        #  define it entirely in tf ops, instead of a placeholder and feeding.
        with tf.device('/cpu:0'):
            lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
            self.summaries.append(tf.summary.scalar('lr', lr))
            return lr

    def optimizer(self):
        logging.info("Optimizer:%s", self._optimization)
        if self._optimization == 'sgd':
            return tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            return optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=args.momentum)
        else:
            logging.error("Invalid optimization flag %s", self._optimization)
            exit(-1)
