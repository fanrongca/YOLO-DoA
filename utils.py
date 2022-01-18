#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : PyCharm
#   File name   : utils.py
#   Author      : Rong Fan
#   Created date: 2021-10-18 12:06:21
#   Description :
#
#================================================================

import numpy as np


def bboxes_iou(boxes1, boxes2):
    """
    Implementation of the iou calculation procedure
    Arguments:
    boxes1 -- the predicted box which has been adopted in the results
    boxes2 -- the predicted box which in the backup set
    Returns:
    the IoU ratio between boxes1 and boxes2
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def nms(bboxes, sigma=0.3):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    best_bboxes = []
    while len(bboxes) > 0:
        max_ind = np.argmax(bboxes[:, 4])
        best_bbox = bboxes[max_ind]
        best_bboxes.append(best_bbox)
        bboxes = np.concatenate([bboxes[: max_ind], bboxes[max_ind + 1:]])
        iou = bboxes_iou(best_bbox[np.newaxis, :4], bboxes[:, :4])
        weight = np.exp(-(1.0 * iou ** 2 / sigma))
        bboxes[:, 4] = bboxes[:, 4] * weight
        score_mask = bboxes[:, 4] > 0.
        bboxes = bboxes[score_mask]

    return best_bboxes


def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):
    """
    Implementation of the decode procedure

    Arguments:
    pred_bbox -- input tensor of shape (BATCH_SIZE, n_H_prev, n_W_prev, n_C_prev)
    org_img_shape -- integer, specifying the shape of original sample
    input_size -- integer, specifying the shape of input sample
    score_threshold -- The predicted values with confidence lower than the threshold are filtered out
    Returns:
    output -- the true coordinate of angular boxes with confidence scores
    """
    valid_scale = [0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape, org_img_shape
    width_resize_ratio = input_size / org_w
    height_resize_ratio = input_size / org_h

    pred_coor[:, 0::2] = 1.0 * pred_coor[:, 0::2] / width_resize_ratio
    pred_coor[:, 1::2] = 1.0 * pred_coor[:, 1::2] / height_resize_ratio

    # # (3) clip boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # # (4) discard invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard boxes with low scores
    scores = pred_conf
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores = pred_coor[mask], scores[mask]

    return np.concatenate([coors, scores[:, np.newaxis]], axis=-1)



