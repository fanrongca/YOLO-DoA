#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2021 * Ltd. All rights reserved.
#
#   Editor      : PyCharm
#   File name   : computationcost.py
#   Author      : Rong Fan
#   Created date: 2021-10-18 12:06:21
#   Description :
#
#================================================================

import tensorflow as tf
from yolovdoa import YOLODOA
BATCH_SIZE = 200
input_shape = 192


def count_flops(graph):
    """
    Implementation for calculation of parameters and computational cost
    Noted that
    Arguments:
    graph -- the whole structure of YOLO-DoA
    Returns:
    """
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('GFLOPs: {};    Trainable params: {}'.format(flops.total_float_ops/1000000000.0, params.total_parameters))
    pass


keep_prob = tf.placeholder(tf.float32, name="keep_prob")
is_training = tf.placeholder(dtype=tf.bool, name='is_trainning')
tf_input_data = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, input_shape, 1, 1], name='input_data')
trainable = tf.placeholder(dtype=tf.bool, name='training')
yolo_model = YOLODOA(tf_input_data, trainable)
sess = tf.Session()
count_flops(sess.graph)
pass
