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
    def __init__(self, stage, croplen, nclasses, optimization=None, momentum=None, reuse_variable=False):
        self.stage = stage
        self.croplen = croplen
        self.nclasses = nclasses
        self.dataloader = None

        self._optimization = optimization
        self._momentum = momentum
        self._train = None
        self._reuse = reuse_variable

        # Touch to initialize
        # if optimization:
        #     self.learning_rate
        #     self.global_step
        #     self.optimizer

    def create_dataloader(self, db_path):
        self.dataloader.stage = self.stage
        self.dataloader.croplen = self.croplen
        self.dataloader.nclasses = self.nclasses

    @model_property
    def train(self):
        return self._train


    @model_property
    def global_step(self):
        # Force global_step on the CPU, becaues the GPU's first step will end at 0 instead of 1.
        with tf.device('/cpu:0'):
            return tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                   trainable=False)

    @model_property
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
        elif self._optimization == 'adadelta':
            return tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
        elif self._optimization == 'adagrad':
            return tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
        elif self._optimization == 'adagradda':
            return tf.train.AdagradDAOptimizer(learning_rate=self.learning_rate,
                                               global_step=self.global_step)
        elif self._optimization == 'momentum':
            return tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                              momentum=self._momentum)
        elif self._optimization == 'adam':
            return tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self._optimization == 'ftrl':
            return tf.train.FtrlOptimizer(learning_rate=self.learning_rate)
        elif self._optimization == 'rmsprop':
            return tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                                             momentum=self._momentum)
        else:
            logging.error("Invalid optimization flag %s", self._optimization)
            exit(-1)

    def get_tower_losses(self, tower):
        """
        Return list of losses

        If user-defined model returns only one loss then this is encapsulated into
        the expected list of dicts structure
        """

        if isinstance(tower.loss, list):
            return tower.loss
        else:
            return [{'loss': tower.loss, 'vars': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)}]


class Tower(object):

    def __init__(self, x, y, input_shape, nclasses, is_training, is_inference):
        self.input_shape = input_shape
        self.nclasses = nclasses
        self.is_training = is_training
        self.is_inference = is_inference
        self.summaries = []
        self.x = x
        self.y = y
        self.train = None

    def gradientUpdate(self, grad):
        return grad
