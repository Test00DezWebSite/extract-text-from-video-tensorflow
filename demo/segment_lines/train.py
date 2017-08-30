#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import argparse
import os
import math
import json
import pdb

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

from cfgs.config import cfg

BATCH_SIZE = cfg.batch_size

class Model(ModelDesc):
    def __init__(self):
        self.channels = cfg.channels
        self.channels.append(cfg.class_num)

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, None, None, 1], 'input'),
                InputDesc(tf.int32, [None, None], 'label')
               ]

    def _build_graph(self, input_vars):
        image, label = input_vars
        image = tf.identity(image, name="NETWORK_INPUT")
        tf.summary.image('input-image', image, max_outputs=5)
        l = image / 255.0 * 2 - 1

        with tf.variable_scope('segmentation') as scope:
            for layer_idx, dilation in enumerate(cfg.dilations):
                layer_input = tf.identity(l)
                if dilation == 1:
                    l = Conv2D('conv.{}'.format(layer_idx),
                               l,
                               self.channels[layer_idx],
                               (cfg.kernel_size[layer_idx], cfg.kernel_size[layer_idx]),
                               'SAME',
                               use_bias=not cfg.with_bn)
                else:
                    l = AtrousConv2D('atrous_conv.{}'.format(layer_idx),
                                     l,
                                     dilation,
                                     self.channels[layer_idx],
                                     (cfg.kernel_size[layer_idx], cfg.kernel_size[layer_idx]),
                                     'SAME',
                                     use_bias=not cfg.with_bn,
                                     mannual_atrous=False)

                if cfg.with_bn == True:
                    l = BatchNorm('bn.{}'.format(layer_idx), l)

                if layer_idx == len(cfg.dilations) - 1:
                    l = l
                else:
                    l = tf.nn.relu(l)

        output = tf.identity(l, name="NETWORK_OUTPUT")
        softmax_output = tf.nn.softmax(output, name="softmax_output")

        label = tf.cast(label, tf.int32)
        label = label - 1
        label_indicator = tf.greater(label, -1)
        effective_label = tf.boolean_mask(tensor=label,
                                          mask=label_indicator)

        output = tf.reshape(output, [BATCH_SIZE, -1, cfg.class_num])
        effective_output = tf.boolean_mask(tensor=output,
                                           mask=label_indicator)

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=effective_output,
            labels=effective_label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = prediction_incorrect(effective_output, effective_label)
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W
        if cfg.weight_decay > 0:
            wd_cost = tf.multiply(cfg.weight_decay, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
            add_moving_summary(cost, wd_cost)

            self.cost = tf.add_n([cost, wd_cost], name='cost')
        else:
            add_moving_summary(cost)
            self.cost = tf.identity(cost, name='cost')

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.1, summary=True)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
