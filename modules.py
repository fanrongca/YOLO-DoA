#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2021. All rights reserved.
#
#   Author      : Rong Fan
#   Created date: 2021-11-18 12:06:21
#   Description : Implementation of the CBL,Ups,SPP Module as defined in Figure 2 of the paper
#
# ================================================================

import tensorflow as tf


def CBL(input, kernel_size, is_training, name, is_downsampling=False, is_activation=True, is_bn=True):
    """
        Implementation of the CBL Module as defined in Figure 2 of the paper

        Arguments:
        input -- the input sample of shape
        kernel_size -- the kernel size of convolutional layer
        is_training -- boolean, specifying the status of training or prediction
        name -- the name of this CBL Module
        is_downsampling-- boolean, specifying the status of downsampling
        is_activation -- boolean, specifying the status of activation function
        is_bn -- boolean, specifying the status of  batch normalization
        Returns:
        output--the feature maps after upsampling
        """

    if is_downsampling:
        strides = (1, 2, 1, 1)
        padding = 'SAME'
    else:
        strides = (1, 1, 1, 1)
        padding = "SAME"

    weight = tf.get_variable(name=name + '_weight', dtype=tf.float32, trainable=True,
                             shape=kernel_size, initializer=tf.random_normal_initializer(stddev=0.01))
    conv = tf.nn.conv2d(input=input, filter=weight, strides=strides, padding=padding, name=name + '_inner_conv1')

    if is_bn:
        conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                             gamma_initializer=tf.ones_initializer(),
                                             moving_mean_initializer=tf.zeros_initializer(),
                                             moving_variance_initializer=tf.ones_initializer(), training=is_training,
                                             name=name + '_inner_conv2')
    else:
        bias = tf.get_variable(name=name + '_bias', shape=kernel_size[-1], trainable=True,
                               dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, bias)

    if is_activation == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv


def Ups(input, name):
    """
        Implementation of the upsampling(Ups)as defined in Figure 2 of the paper

        Arguments:
        input -- the input sample of shape
        name -- the name of this UPS Module
        method -- the type of the upsample
        Returns:
        output--the feature maps after upsampling
        """

    input_shape = tf.shape(input)
    output = tf.image.resize_nearest_neighbor(input, (input_shape[1] * 2, 1), name='Ups' + name)

    return output

pass


# spp layer
def spp_layer(x):
    """
        Implementation of the Spatial Pyramid Pooling(SPP)

        Arguments:
        input -- the input sample of shape
        Returns:
        output--the feature maps after SPP
        """
    x1 = x
    x2 = tf.layers.max_pooling2d(x, pool_size=(2, 1), strides=(1, 1), padding='same')
    x3 = tf.layers.max_pooling2d(x, pool_size=(3, 1), strides=(1, 1), padding='same')
    x4 = tf.layers.max_pooling2d(x, pool_size=(5, 1), strides=(1, 1), padding='same')
    result = tf.concat([x1, x2, x3, x4], axis=-1)
    return result

pass