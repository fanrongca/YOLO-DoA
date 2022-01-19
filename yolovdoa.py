#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2021. All rights reserved.
#
#   Author      : Rong Fan
#   Created date: 2021-11-18 12:06:21
#   Description : Main function of YOLO-DoA, including Neck, Head sturcture, and loss calculation
#
# ================================================================

import numpy as np
import tensorflow as tf
import modules
import backbone


class YOLODOA(object):
    def __init__(self, input, is_training, iou_loss=True, spp_layer=False, grid_sensitive=False):

        self.is_training        = is_training
        self.strides          = [8]  # the range of each SubReg (△)
        self.microreg_number = 3  # the numerber of MicroRegs (P)
        self.iouloss_threshold  = 0.5  # the threshold value of soft-NMS
        self.iou_loss = iou_loss # boolean, status of using iou_loss
        self.spp_layer = spp_layer # boolean, status of using spp_layer
        self.grid_sensitive = grid_sensitive # boolean, status of using grid_sensitive

        try:
            self.predicted_offsets = self.__build_network(input)
        except:
            raise NotImplementedError("Network build failed!")

        self.predicted_boxes = self.transform(self.predicted_offsets, self.strides[0])

    def __build_network(self, input_x):
        """
            Implementation of the Neck and Head as defined in Figure 2 of the paper

            Arguments:
            input_x -- the input_x sampe of shape

            Returns:
            output--the output feature maps of YOLO-DoA
            """
        # the output feature maps of BackBone, and then are fused in the Neck
        features_24_1, features_12_1, input_x = backbone.CSP_SE_ResNet18vd(input_x, self.is_training)

        # the beginning of first CBF Module in Fig.2
        input_x = modules.CBL(input_x, (1, 1, 64, 32), self.is_training, 'layer52')
        in_c = 32
        input_c = in_c

        input_x = modules.CBL(input_x, (3, 1, input_c, 64), self.is_training, 'layer53')
        input_x = modules.CBL(input_x, (1, 1, 64, in_c), self.is_training, 'layer54')
        # spp layer
        if self.spp_layer:
            input_x = modules.spp_layer(input_x)
            input_x = modules.CBL(input_x, (1, 1, input_c * 4, 64), self.is_training, 'layer_spp1')
            input_x = modules.CBL(input_x, (3, 1, 64, input_c * 2), self.is_training, 'layer_spp2')

            temp_inputc = input_c*2
            input_x = modules.CBL(input_x, (3, 1, temp_inputc, 64), self.is_training, 'layer55_spp')
            input_x = modules.CBL(input_x, (1, 1, 64, in_c), self.is_training, 'layer56_spp')
        else:
            input_x = modules.CBL(input_x, (3, 1, input_c, 64), self.is_training, 'layer55')
            input_x = modules.CBL(input_x, (1, 1, 64, in_c), self.is_training, 'layer56')
        # the end of first CBF Module in Fig.2

        # the beginning of first UPS Module in Fig.2
        input_x = modules.CBL(input_x, (1, 1, input_c, 16), self.is_training, 'layer57')
        input_x = modules.Ups(input_x, name='upsample0')
        # the end of first UPS Module in Fig.2

        # the first concatenation operation
        with tf.variable_scope('concatenation_operation_1'):
            input_x = tf.concat([input_x, features_12_1], axis=-1)

        # the beginning of second CBF Module in Fig.2
        input_x = modules.CBL(input_x, (1, 1, 80, 32), self.is_training, 'layer58')
        in_c = 32
        input_c = in_c
        input_x = modules.CBL(input_x, (3, 1, input_c, 64), self.is_training, 'layer59')
        input_x = modules.CBL(input_x, (1, 1, 64, in_c), self.is_training, 'layer60')

        input_x = modules.CBL(input_x, (3, 1, input_c, 64), self.is_training, 'layer61')
        input_x = modules.CBL(input_x, (1, 1, 64, in_c), self.is_training, 'layer62')
        # the end of second CBF Module in Fig.2

        # the beginning of second UPS Module in Fig.2
        input_x = modules.CBL(input_x, (1, 1, input_c, 16), self.is_training, 'layer63')
        input_x = modules.Ups(input_x, name='upsample1')
        # the end of second UPS Module in Fig.2

        # the second concatenation operation
        with tf.variable_scope('concatenation_operation_2'):
            input_x = tf.concat([input_x, features_24_1], axis=-1)

        # the beginning of third CBF Module in Fig.2
        input_x = modules.CBL(input_x, (1, 1, 48, 16), self.is_training, 'layer64')
        in_c = 16
        input_c = in_c
        input_x = modules.CBL(input_x, (3, 1, input_c, 32), self.is_training, 'layer65')
        input_x = modules.CBL(input_x, (1, 1, 32, in_c), self.is_training, 'layer66')
        input_x = modules.CBL(input_x, (3, 1, input_c, 32), self.is_training, 'layer67')
        input_x = modules.CBL(input_x, (1, 1, 32, in_c), self.is_training, 'layer68')
        # the end of third CBF Module in Fig.2

        # the beginning of Head Module
        conv_sobj_branch = modules.CBL(input_x, (3, 1, input_c, 32), self.is_training, name='conv_obj_branch')
        sobj_in_c = 32
        sobj_input_c = sobj_in_c
        conv_box = modules.CBL(conv_sobj_branch, (1, 1, sobj_input_c, 3 * 5),
                               is_training=self.is_training, name='predicted_offsets', is_activation=False, is_bn=False)
        # the end of Head Module

        # the output feature maps of Neck, which can be used to transform to locations of DoAs
        return conv_box

    def transform(self, conv_output, stride):
        """
        transform the output of YOLO-Doa into the true angular coordinates
        return tensor of shape [BATCH_SIZE, 24, 1, 3, 5 ]
               contains (x, y, w, h, score)
        """

        conv_shape       = tf.shape(conv_output)
        batch_size       = conv_shape[0]
        output_size      = conv_shape[1]
        number_microreg = 3
        microreg = [0.25, 0.125]
        conv_output = tf.reshape(conv_output, (batch_size, output_size, 1, number_microreg, 5))

        conv_raw_dx = conv_output[:, :, :, :, 0]
        conv_raw_dy = conv_output[:, :, :, :, 1]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]

        if self.grid_sensitive:
            alpha = 1.05
        else:
            alpha = 1.0

        x_grid = tf.range(output_size, dtype=tf.int32)
        x_grid = tf.cast(x_grid, tf.float32)
        x_grid = tf.reshape(x_grid, [1, output_size, 1, 1])
        x_grid = tf.tile(x_grid, [batch_size, 1, 1, 3])
        pred_x = (alpha*tf.sigmoid(conv_raw_dx) + x_grid-(alpha-1)/2) * stride
        pred_y = (alpha*tf.sigmoid(conv_raw_dy) - (alpha-1)/2)
        pred_x = tf.expand_dims(pred_x, axis=-1)
        pred_y = tf.expand_dims(pred_y, axis=-1)

        pred_wh = (tf.exp(conv_raw_dwdh) * microreg) * stride
        pred_xywh = tf.concat([pred_x, pred_y, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)

        return tf.concat([pred_xywh, pred_conf], axis=-1)

    def focal_loss(self, predicted_value, true_label, alpha=1, gamma=2):
        focal_loss = alpha * tf.pow(tf.abs(predicted_value - true_label), gamma)
        return focal_loss

    def box_l2(self, predicted_box, angular_box, input_size):
        """
        The original regression loss of YOLOv3
        """
        width = tf.cast(input_size, tf.float32)
        predicted_box = tf.concat([predicted_box[..., :2] - predicted_box[..., 2:] * 0.5,
                                   predicted_box[..., :2] + predicted_box[..., 2:] * 0.5], axis=-1)
        angular_box = tf.concat([angular_box[..., :2] - angular_box[..., 2:] * 0.5,
                                 angular_box[..., :2] + angular_box[..., 2:] * 0.5], axis=-1)
        mse = tf.square((predicted_box[..., 0] - angular_box[..., 0]) / width) + \
              tf.square(predicted_box[..., 1] - angular_box[..., 1]) + \
              tf.square((predicted_box[..., 2] - angular_box[..., 2]) / width) + \
              tf.square(predicted_box[..., 3] - angular_box[..., 3])
        return mse

    def box_giou(self, predicted_box, angular_box):
        """
        The regression loss with GIoU Loss
        """
        predicted_box = tf.concat([predicted_box[..., :2] - predicted_box[..., 2:] * 0.5,
                                   predicted_box[..., :2] + predicted_box[..., 2:] * 0.5], axis=-1)
        angular_box = tf.concat([angular_box[..., :2] - angular_box[..., 2:] * 0.5,
                                 angular_box[..., :2] + angular_box[..., 2:] * 0.5], axis=-1)

        predicted_box = tf.concat([tf.minimum(predicted_box[..., :2], predicted_box[..., 2:]),
                                   tf.maximum(predicted_box[..., :2], predicted_box[..., 2:])], axis=-1)
        angular_box = tf.concat([tf.minimum(angular_box[..., :2], angular_box[..., 2:]),
                                 tf.maximum(angular_box[..., :2], angular_box[..., 2:])], axis=-1)

        predictedbox_area = (predicted_box[..., 2] - predicted_box[..., 0]) * (predicted_box[..., 3] - predicted_box[..., 1])
        angularbox_area = (angular_box[..., 2] - angular_box[..., 0]) * (angular_box[..., 3] - angular_box[..., 1])

        leftup_coord = tf.maximum(predicted_box[..., :2], angular_box[..., :2])
        rightdown_coord = tf.minimum(predicted_box[..., 2:], angular_box[..., 2:])

        inter_section = tf.maximum(rightdown_coord - leftup_coord, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = predictedbox_area + angularbox_area - inter_area
        iou = inter_area / (union_area + 1e-6)

        final_left_up = tf.minimum(predicted_box[..., :2], angular_box[..., :2])
        final_right_down = tf.maximum(predicted_box[..., 2:], angular_box[..., 2:])
        final_area = tf.maximum(final_right_down - final_left_up, 0.0)
        final_area = final_area[..., 0] * final_area[..., 1]
        giou = iou - 1.0 * (final_area - union_area) / (final_area + 1e-6)

        return giou

    def box_iou(self, predicted_box, angular_box):
        """
        The IOU ratio between predicted_box and angular_box
        """
        predictedbox_area = predicted_box[..., 2] * predicted_box[..., 3]
        angularbox_area = angular_box[..., 2] * angular_box[..., 3]

        predicted_box = tf.concat([predicted_box[..., :2] - predicted_box[..., 2:] * 0.5,
                                   predicted_box[..., :2] + predicted_box[..., 2:] * 0.5], axis=-1)
        angular_box = tf.concat([angular_box[..., :2] - angular_box[..., 2:] * 0.5,
                                 angular_box[..., :2] + angular_box[..., 2:] * 0.5], axis=-1)

        leftup_coord = tf.maximum(predicted_box[..., :2], angular_box[..., :2])
        rightdown_coord = tf.minimum(predicted_box[..., 2:], angular_box[..., 2:])

        inter_section = tf.maximum(rightdown_coord - leftup_coord, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = predictedbox_area + angularbox_area - inter_area
        iou = 1.0 * inter_area / union_area

        return iou

    def loss_fuction(self, predicted_offsets, predicted_boxes, label, angular_boxes, stride):
        """
        Implementation of Loss fuction  for Formula 6 of the paper

        Arguments:
        predicted_offsets -- the output feature maps for weighted cross-entropy function (confidence loss)
        predicted_boxes -- the transformed coordinates for GIou Loss (regression loss)
        label -- the ture labels of incident directions
        angular_boxes -- the true angular boxes for incident directions
        stride -- the range of each SubReg

        Returns:
        giou_loss--the regression loss of YOLO-DoA
        conf_loss--the confidence loss of YOLO-DoA
        """
        offset_shape  = tf.shape(predicted_offsets)
        batchsize  = offset_shape[0]
        output_size = offset_shape[1]
        input_size  = stride * output_size
        predicted_offsets = tf.reshape(predicted_offsets, (batchsize, output_size, 1,
                                                           self.microreg_number, 5))
        conf_offest = predicted_offsets[:, :, :, :, 4:5]
        predicted_xywh     = predicted_boxes[:, :, :, :, 0:4]
        predicted_conf     = predicted_boxes[:, :, :, :, 4:5]
        label_xywh    = label[:, :, :, :, 0:4]
        is_responsible  = label[:, :, :, :, 4:5]

        input_size = tf.cast(input_size, tf.float32)
        box_loss_scale = 2.0
        if self.iou_loss:
            giou = tf.expand_dims(self.box_giou(predicted_xywh, label_xywh), axis=-1)
            giou_loss = is_responsible * box_loss_scale * (1 - giou)
        else:
            giou = tf.expand_dims(self.box_l2(predicted_xywh, label_xywh, input_size), axis=-1)
            giou_loss = is_responsible * box_loss_scale * giou

        iou = self.box_iou(predicted_xywh[:, :, :, :, np.newaxis, :], angular_boxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        respond_bgd = (1.0 - is_responsible) * tf.cast(max_iou < self.iouloss_threshold, tf.float32)

        conf_focal = self.focal_loss(is_responsible, predicted_conf)

        conf_loss = conf_focal * (
                is_responsible * tf.nn.sigmoid_cross_entropy_with_logits(labels=is_responsible, logits=conf_offest)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=is_responsible, logits=conf_offest)
        )

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))

        return giou_loss, conf_loss

    def loss_calculation(self, label_box, true_box):
        """
        Implementation of the Loss function
        """
        loss_box = self.loss_fuction(self.predicted_offsets, self.predicted_boxes, label_box, true_box,
                                     stride=self.strides[0])

        giou_loss = loss_box[0]

        confidence_loss = loss_box[1]

        return giou_loss, confidence_loss

    def get_predict_boxs(self):
        """
        return the predicted boxes
        """
        return self.predicted_boxes

pass