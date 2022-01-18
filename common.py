#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2021 * Ltd. All rights reserved.
#
#   Editor      : PyCharm
#   File name   : common.py
#   Author      : Rong Fan
#   Created date: 2021-10-18 12:06:21
#   Description :
#
#================================================================

import tensorflow as tf


def CBL(input_data, filters_shape, trainable, name, downsample=False, activate=True, bn=True):
    """
        Implementation of the CBL Module as defined in Figure 2 of the paper

        Arguments:
        input_data -- the input sample of shape
        filters_shape -- the kernel size of convolutional layer
        trainable -- boolean, specifying the status of training or prediction
        name -- the name of this CBL Module
        downsample-- boolean, specifying the status of downsampling
        activate -- boolean, specifying the status of activation function
        bn -- boolean, specifying the status of  batch normalization
        Returns:
        output--the feature maps after upsampling
        """

    with tf.variable_scope(name):
        if downsample:
            strides = (1, 2, 1, 1)
            padding = 'SAME'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"

        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)

        if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv


def Ups(input_data, name, method="deconv"):
    """
        Implementation of the upsampling(Ups)as defined in Figure 2 of the paper

        Arguments:
        input -- the input sample of shape
        name -- the name of this UPS Module
        method -- the type of the upsample
        Returns:
        output--the feature maps after upsampling
        """
    assert method in ["resize", "deconv"]

    if method == "resize":
        with tf.variable_scope(name):
            input_shape = tf.shape(input_data)
            output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, 1))

    if method == "deconv":
        # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT optimization
        numm_filter = input_data.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(input_data, numm_filter, kernel_size=(2, 1), padding='same',
                                            strides=(2, 1), kernel_initializer=tf.random_normal_initializer())

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
