#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : PyCharm
#   File name   : yolovdoa.py
#   Author      : Rong Fan
#   Created date: 2021-10-18 12:06:21
#   Description :
#
#================================================================

import numpy as np
import tensorflow as tf
import common as common
import backbone as backbone

iou_loss = True
spp_layer = False
grid_sensitive = False


class YOLODOA(object):
    """Implement tensoflow YOLODOA here"""
    def __init__(self, input_data, trainable):

        self.trainable        = trainable
        self.strides          = [8]  # the range of each SubReg (△)
        self.microreg_per_scale = 3  # the numerber of MicroRegs (P)
        self.iou_loss_thresh  = 0.5  # the threshold value of soft-NMS
        self.upsample_method  = "resize"

        try:
            self.conv_box = self.__build_nework(input_data)
        except:
            raise NotImplementedError("Can not build up yolo-doa network!")

        with tf.variable_scope('pred_box'):
            self.pred_box = self.decode(self.conv_box, self.strides[0])

    def __build_nework(self, input_data):
        """
            Implementation of the Neck and Head as defined in Figure 2 of the paper

            Arguments:
            input_data -- the input sampe of shape

            Returns:
            output--the output feature maps of YOLO-DoA
            """
        # the output feature maps of BackBone, and then are fused in the Neck
        route_1, route_2, input_data = backbone.CSP_SE_ResNet18vd(input_data, self.trainable)

        # the beginning of first CBF Module in Fig.2
        input_data = common.CBL(input_data, (1, 1, 64, 32), self.trainable, 'conv52')
        in_c = 32
        input_c = in_c

        input_data = common.CBL(input_data, (3, 1, input_c, 64), self.trainable, 'conv53')
        input_data = common.CBL(input_data, (1, 1, 64, in_c), self.trainable, 'conv54')
        # spp layer
        if spp_layer:
            input_data = common.spp_layer(input_data)
            input_data = common.CBL(input_data, (1, 1, input_c * 4, 64), self.trainable, 'conv_spp1')
            input_data = common.CBL(input_data, (3, 1, 64, input_c * 2), self.trainable, 'conv_spp2')

            temp_inputc = input_c*2
            input_data = common.CBL(input_data, (3, 1, temp_inputc, 64), self.trainable, 'conv55_spp')
            input_data = common.CBL(input_data, (1, 1, 64, in_c), self.trainable, 'conv56_spp')
        else:
            input_data = common.CBL(input_data, (3, 1, input_c, 64), self.trainable, 'conv55')
            input_data = common.CBL(input_data, (1, 1, 64, in_c), self.trainable, 'conv56')
        # the end of first CBF Module in Fig.2

        # the beginning of first UPS Module in Fig.2
        input_data = common.CBL(input_data, (1, 1, input_c, 16), self.trainable, 'conv57')
        input_data = common.Ups(input_data, name='upsample0', method=self.upsample_method)
        # the end of first UPS Module in Fig.2

        # the first concatenation operation
        with tf.variable_scope('route_1'):
            input_data = tf.concat([input_data, route_2], axis=-1)

        # the beginning of second CBF Module in Fig.2
        input_data = common.CBL(input_data, (1, 1, 80, 32), self.trainable, 'conv58')
        in_c = 32
        input_c = in_c
        input_data = common.CBL(input_data, (3, 1, input_c, 64), self.trainable, 'conv59')
        input_data = common.CBL(input_data, (1, 1, 64, in_c), self.trainable, 'conv60')

        input_data = common.CBL(input_data, (3, 1, input_c, 64), self.trainable, 'conv61')
        input_data = common.CBL(input_data, (1, 1, 64, in_c), self.trainable, 'conv62')
        # the end of second CBF Module in Fig.2

        # the beginning of second UPS Module in Fig.2
        input_data = common.CBL(input_data, (1, 1, input_c, 16), self.trainable, 'conv63')
        input_data = common.Ups(input_data, name='upsample1', method=self.upsample_method)
        # the end of second UPS Module in Fig.2

        # the second concatenation operation
        with tf.variable_scope('route_2'):
            input_data = tf.concat([input_data, route_1], axis=-1)

        # the beginning of third CBF Module in Fig.2
        input_data = common.CBL(input_data, (1, 1, 48, 16), self.trainable, 'conv64')
        in_c = 16
        input_c = in_c
        input_data = common.CBL(input_data, (3, 1, input_c, 32), self.trainable, 'conv65')
        input_data = common.CBL(input_data, (1, 1, 32, in_c), self.trainable, 'conv66')
        input_data = common.CBL(input_data, (3, 1, input_c, 32), self.trainable, 'conv67')
        input_data = common.CBL(input_data, (1, 1, 32, in_c), self.trainable, 'conv68')
        # the end of third CBF Module in Fig.2

        # the beginning of Head Module
        conv_sobj_branch = common.CBL(input_data, (3, 1, input_c, 32), self.trainable, name='conv_obj_branch')
        sobj_in_c = 32
        sobj_input_c = sobj_in_c
        conv_box = common.CBL(conv_sobj_branch, (1, 1, sobj_input_c, 3 * 5),
                              trainable=self.trainable, name='conv_box', activate=False, bn=False)
        # the end of Head Module

        # the output feature maps of Neck, which can be used to transform to locations of DoAs
        return conv_box

    def decode(self, conv_output, stride):
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

        if grid_sensitive:
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

    def focal(self, target, actual, alpha=1, gamma=2):
        focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss

    def bbox_l2(self, boxes1, boxes2, input_size):
        """
        The original regression loss of YOLOv3
        """
        width = tf.cast(input_size, tf.float32)
        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
        mse = tf.square((boxes1[..., 0] - boxes2[..., 0])/width) + \
              tf.square(boxes1[..., 1] - boxes2[..., 1]) + \
              tf.square((boxes1[..., 2] - boxes2[..., 2])/width) + \
              tf.square(boxes1[..., 3] - boxes2[..., 3])
        return mse

    def bbox_giou(self, boxes1, boxes2):
        """
        The regression loss with GIoU Loss
        """
        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / (union_area + 1e-6)
        # added 1e-6 in denominator to avoid generation of inf, which may cause nan loss

        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        giou = iou - 1.0 * (enclose_area - union_area) / (enclose_area + 1e-6)
        # added 1e-6 in denominator to avoid generation of inf, which may cause nan loss

        return giou

    def bbox_iou(self, boxes1, boxes2):
        """
        The IOU ratio between boxes1 and boxes2
        """
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = 1.0 * inter_area / union_area

        return iou

    def loss_layer(self, conv, pred, label, bboxes, stride):
        """
        Implementation of Loss fuction  for Formula 6 of the paper

        Arguments:
        conv -- the output feature maps for weighted cross-entropy function (confidence loss)
        pred -- the transformed coordinates for GIou Loss (regression loss)
        label -- the ture labels of incident directions
        stride -- the range of each SubReg

        Returns:
        giou_loss--the regression loss of YOLO-DoA
        conf_loss--the confidence loss of YOLO-DoA
        """
        conv_shape  = tf.shape(conv)
        batch_size  = conv_shape[0]
        output_size = conv_shape[1]
        input_size  = stride * output_size
        conv = tf.reshape(conv, (batch_size, output_size, 1,
                                 self.microreg_per_scale, 5))
        conv_raw_conf = conv[:, :, :, :, 4:5]
        pred_xywh     = pred[:, :, :, :, 0:4]
        pred_conf     = pred[:, :, :, :, 4:5]
        label_xywh    = label[:, :, :, :, 0:4]
        respond_bbox  = label[:, :, :, :, 4:5]

        input_size = tf.cast(input_size, tf.float32)
        bbox_loss_scale = 2.0
        if iou_loss:
            giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)
            giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)
        else:
            giou = tf.expand_dims(self.bbox_l2(pred_xywh, label_xywh, input_size), axis=-1)
            giou_loss = respond_bbox * bbox_loss_scale * giou

        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.iou_loss_thresh, tf.float32)

        conf_focal = self.focal(respond_bbox, pred_conf)

        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))

        return giou_loss, conf_loss

    def compute_loss(self, label_box, true_box):
        """
        Implementation of the Loss function
        """
        with tf.name_scope('box_loss'):
            loss_box = self.loss_layer(self.conv_box, self.pred_box, label_box, true_box,
                                         stride=self.strides[0])

        with tf.name_scope('giou_loss'):
            giou_loss = loss_box[0]

        with tf.name_scope('conf_loss'):
            conf_loss = loss_box[1]

        return giou_loss, conf_loss

    def get_predict_boxs(self):
        """
        return the predicted boxes
        """
        return self.pred_box

pass