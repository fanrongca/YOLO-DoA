#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : PyCharm
#   File name   : backbone.py
#   Author      : Rong Fan
#   Created date: 2021-10-18 12:06:21
#   Description :
#
#================================================================

import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
use_SE_Block = True
BATCH_SIZE = 200
output_shape = 180


def CSP_SE_ResNet18vd(input, is_training):
    """
    Implementation of the Backbone as defined in Figure 2 of the paper

    Arguments:
    input -- the input sample of shape (BATCH_SIZE, 192, 1, 1)
    is_training -- boolean, specifying the status of training or prediction

    Returns:
    x_24_1 -- output of the backbone, tensor of shape (BATCH_SIZE, 24, 1, 3, 5)
    x_12_1 -- output of the backbone, tensor of shape (BATCH_SIZE, 12, 1, 3, 5)
    x_6_1 -- output of the backbone, tensor of shape (BATCH_SIZE, 6, 1, 3, 5)
    """

    # stage 1
    # the first CBR Module
    x = tf.layers.conv2d(input, filters=8, kernel_size=(7, 1), strides=(1, 1), name='conv1', padding='same')
    x = tf.layers.batch_normalization(x, axis=3, name='bn_conv1', training=is_training)
    x = tf.nn.relu(x)
    # the maxpooling step
    x = tf.layers.max_pooling2d(x, pool_size=(3, 1), strides=(2, 1), padding='same')
    # x,[?,96,1,8]

    # stage 2
    # the fist CSR Module with input channel=8 and output channel=16
    downsample_in3x3 = True
    shortcut = tf.layers.conv2d(x, filters=4, kernel_size=(1, 1), strides=(2, 1), name='conv_stage2_1')
    shortcut = tf.layers.batch_normalization(shortcut, axis=3, name='bn_stage2_conv1', training=is_training)
    shortcut = tf.nn.relu(shortcut)

    shortcut = tf.layers.conv2d(shortcut, filters=4, kernel_size=(1, 1), strides=(1, 1),
                                       name='conv_stage2_1_2')
    shortcut = tf.layers.batch_normalization(shortcut, axis=3, name='bn_stage2_conv1_2', training=is_training)
    shortcut = tf.nn.relu(shortcut)

    x = tf.layers.conv2d(x, filters=4, kernel_size=(1, 1), strides=(1, 1), name='conv_stage2_2')
    x = tf.layers.batch_normalization(x, axis=3, name='bn_stage2_conv2', training=is_training)
    x = tf.nn.relu(x)

    x = convolutional_block(x, kernel_size=3, filters=[4, 4, 4], stage=2, block='a', stride=2, training=is_training,
                            downsample_in3x3=downsample_in3x3, is_first=True)
    x = identity_block(x, 3, [4, 4, 4], stage=2, block='b', training=is_training)
    x = tf.concat([x, shortcut], axis=-1)

    x = tf.layers.conv2d(x, filters=16, kernel_size=(1, 1), strides=(1, 1), name='conv_stage2_2_2')
    x = tf.layers.batch_normalization(x, axis=3, name='bn_stage2_conv2_2', training=is_training)
    x = tf.nn.relu(x)
    # x,[?,48,1,16]
    # end of first CSR Module

    # stage 3  # the second CSR Module with input channel=16 and output channel=32
    shortcut = tf.layers.conv2d(x, filters=8, kernel_size=(1, 1), strides=(2, 1), name='conv_stage3_1')
    shortcut = tf.layers.batch_normalization(shortcut, axis=3, name='bn_stage3_conv1', training=is_training)
    shortcut = tf.nn.relu(shortcut)

    shortcut = tf.layers.conv2d(shortcut, filters=8, kernel_size=(1, 1), strides=(1, 1),
                                       name='conv_stage3_1_2')
    shortcut = tf.layers.batch_normalization(shortcut, axis=3, name='bn_stage3_conv1_2', training=is_training)
    shortcut = tf.nn.relu(shortcut)

    x = tf.layers.conv2d(x, filters=8, kernel_size=(1, 1), strides=(1, 1), name='conv_stage3_2')
    x = tf.layers.batch_normalization(x, axis=3, name='bn_stage3_conv2', training=is_training)
    x = tf.nn.relu(x)

    x = convolutional_block(x, kernel_size=3, filters=[8, 8, 8],
                                            stage=3, block='a', stride=2, training=is_training,
                            downsample_in3x3=downsample_in3x3)
    x = identity_block(x, 3, [8, 8, 8], stage=3, block='b', training=is_training)

    x = tf.concat([x, shortcut], axis=-1)

    x = tf.layers.conv2d(x, filters=32, kernel_size=(1, 1), strides=(1, 1), name='conv_stage3_2_2')
    x = tf.layers.batch_normalization(x, axis=3, name='bn_stage3_conv2_2', training=is_training)
    x = tf.nn.relu(x)
    x_24_1 = x
    # x_24_1,[?,24,1,32]
    # end of second CSR Module

    # stage 4  # the third CSR Module with input channel=32 and output channel=64
    shortcut = tf.layers.conv2d(x, filters=16, kernel_size=(1, 1), strides=(2, 1), name='conv_stage4_1')
    shortcut = tf.layers.batch_normalization(shortcut, axis=3, name='bn_stage4_conv1', training=is_training)
    shortcut = tf.nn.relu(shortcut)

    shortcut = tf.layers.conv2d(shortcut, filters=16, kernel_size=(1, 1), strides=(1, 1),
                                       name='conv_stage4_1_2')
    shortcut = tf.layers.batch_normalization(shortcut, axis=3, name='bn_stage4_conv1_2', training=is_training)
    shortcut = tf.nn.relu(shortcut)

    x = tf.layers.conv2d(x, filters=16, kernel_size=(1, 1), strides=(1, 1), name='conv_stage4_2')
    x = tf.layers.batch_normalization(x, axis=3, name='bn_stage4_conv2', training=is_training)
    x = tf.nn.relu(x)

    x = convolutional_block(x, kernel_size=3, filters=[16, 16, 16], stage=4, block='a', stride=2,
                            training=is_training,
                            downsample_in3x3=downsample_in3x3)
    x = identity_block(x, 3, [16, 16, 16], stage=4, block='b', training=is_training)
    x = tf.concat([x, shortcut], axis=-1)

    x = tf.layers.conv2d(x, filters=64, kernel_size=(1, 1), strides=(1, 1), name='conv_stage4_2_2')
    x = tf.layers.batch_normalization(x, axis=3, name='bn_stage4_conv2_2', training=is_training)
    x = tf.nn.relu(x)
    x_12_1 = x
    # x_12_1,[?,12,1,64]
    # end of third CSR Module

    # stage 5 # the fourth CSR Module with input channel=64 and output channel=64
    shortcut = tf.layers.conv2d(x, filters=32, kernel_size=(1, 1), strides=(2, 1), name='conv_stage5_1')
    shortcut = tf.layers.batch_normalization(shortcut, axis=3, name='bn_stage5_conv1', training=is_training)
    shortcut = tf.nn.relu(shortcut)

    shortcut = tf.layers.conv2d(shortcut, filters=32, kernel_size=(1, 1), strides=(1, 1),
                                       name='conv_stage5_1_2')
    shortcut = tf.layers.batch_normalization(shortcut, axis=3, name='bn_stage5_conv1_2', training=is_training)
    shortcut = tf.nn.relu(shortcut)

    x = tf.layers.conv2d(x, filters=32, kernel_size=(1, 1), strides=(1, 1), name='conv_stage5_2')
    x = tf.layers.batch_normalization(x, axis=3, name='bn_stage5_conv2', training=is_training)
    x = tf.nn.relu(x)

    x = convolutional_block(x, kernel_size=3,filters=[32, 32, 32], stage=5, block='a', stride=2,
                            training=is_training, downsample_in3x3=downsample_in3x3)
    x = identity_block(x, 3, [32, 32, 32], stage=5, block='b', training=is_training)

    x = tf.concat([x, shortcut], axis=-1)

    x = tf.layers.conv2d(x, filters=64, kernel_size=(1, 1), strides=(1, 1), name='conv_stage5_2_2')
    x = tf.layers.batch_normalization(x, axis=3, name='bn_stage5_conv2_2', training=is_training)
    x_6_1 = tf.nn.relu(x)
    # x,[?,6,1,64]
    # end of fourth CSR Module

    # the output of Backbone, which can be used in the Neck
    return x_24_1, x_12_1, x_6_1


def identity_block(X_input, kernel_size, filters, stage, block, training=True):
    """
    Implementation of the residual module with the SE operation (SR) as defined in Figure 2 of the paper

    Arguments:
    X -- input tensor of shape (BATCH_SIZE, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (BATCH_SIZE,n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    with tf.name_scope("id_block_stage"+str(stage)):
        filter1, filter2, filter3 = filters
        X_shortcut = X_input

        # Second component of main path
        x = tf.layers.conv2d(X_input, filter2, (kernel_size, 1),
                                 padding='same', name=conv_name_base+'2b')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base+'2b', training=training)
        x = tf.nn.relu(x)

        # Third component of main path
        x = tf.layers.conv2d(x, filter3, kernel_size=(kernel_size, 1),name=conv_name_base+'2c', padding='same')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2c', training=training)

        if use_SE_Block:
            x = squeeze_excitation_block(x, filter3, int(filter1/4), conv_name_base+'se')

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X_add_shortcut = tf.add(x, X_shortcut)
        add_result = tf.nn.relu(X_add_shortcut)

    return add_result


def convolutional_block(X_input, kernel_size, filters, stage, block, stride=2, training=True, is_first=False,
                        downsample_in3x3=True):
    """
    Implementation of the residual module with the SE operation (SR) as defined in Figure 2 of the paper

    Arguments:
    X -- input tensor of shape (BATCH_SIZE, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    stride -- Integer, specifying the stride to be used

    Returns:
    X -- output of the residual module, tensor of shape (BATCH_SIZE,n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    with tf.name_scope("conv_block_stage" + str(stage)):

        # Retrieve Filters
        filter1, filter2, filter3 = filters

        if downsample_in3x3:
            stride1, stride2 = 1, stride
        else:
            stride1, stride2 = stride, 1

        # Save the input value
        X_shortcut = X_input

        # Second component of main path
        x = tf.layers.conv2d(X_input, filter2, (kernel_size, 1), strides=(stride2, 1), name=conv_name_base + '2b',padding='same')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2b', training=training)
        x = tf.nn.relu(x)

        # Third component of main path
        x = tf.layers.conv2d(x, filter3, (kernel_size, 1), name=conv_name_base + '2c', padding='same')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2c', training=training)

        if use_SE_Block:
            x = squeeze_excitation_block(x, filter3, int(filter1/4), conv_name_base+'se')

        if not is_first:
            X_shortcut = tf.layers.average_pooling2d(X_shortcut, pool_size=(2, 1), strides=(2, 1), padding='same')
            X_shortcut = tf.layers.conv2d(X_shortcut, filter3, (1, 1),
                                      strides=(1, 1), name=conv_name_base + '1_0', padding='same')
            X_shortcut = tf.layers.batch_normalization(X_shortcut, axis=3, name=bn_name_base + '1_0',
                                                       training=training)
        else:
            X_shortcut = tf.layers.conv2d(X_shortcut, filter3, (1, 1),
                                      strides=(stride, 1), name=conv_name_base + '1_1', padding='same')
            X_shortcut = tf.layers.batch_normalization(X_shortcut, axis=3, name=bn_name_base + '1_1', training=training)
        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X_add_shortcut = tf.add(X_shortcut, x)
        add_result = tf.nn.relu(X_add_shortcut)

    return add_result


def squeeze_excitation_block(input_x, out_dim, ratio, stage):
    """
    # Implementation of the squeeze-and-excitation (SE) operation as defined in Figure 2 of the paper

    Arguments:
    input_x -- input tensor of shape (BATCH_SIZE, n_H_prev, n_W_prev, n_C_prev)
    out_dim -- integer, specifying the shape of the output
    ratio -- integer, control the ratio of squeeze operation
    stage -- integer, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the SR Module, tensor of shape (BATCH_SIZE,n_H, n_W, n_C)
    """
    layer_name = "se_block_stage" + str(stage)
    with tf.name_scope(layer_name):
        squeeze = global_avg_pool(input_x)
        excitation = tf.layers.dense(squeeze, use_bias=True, units=out_dim / ratio, name=layer_name + '_fully_connected1')
        excitation = tf.nn.relu(excitation)
        excitation = tf.layers.dense(excitation, use_bias=True, units=out_dim, name=layer_name + '_fully_connected2')
        excitation = tf.nn.sigmoid(excitation)
        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
        scale = input_x * excitation
        return scale
        pass
    pass
