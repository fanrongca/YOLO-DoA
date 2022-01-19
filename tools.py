#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2021. All rights reserved.
#
#   Author      : Rong Fan
#   Created date: 2021-11-18 12:06:21
#   Description : Including data preprocessing, soft-nms, prediction frame filtering and other functions
#
# ================================================================

import numpy as np
microreg_number = 3
max_bbox_per_scale = 2


def preprocess_true_boxes(bboxes, train_output_sizes, strides):
    """
    Implementation of Data Preprocess in the paper
    Arguments:
    bboxes -- the original boxes for incident directions
    train_output_sizes -- the number of SubRegs
    strides -- the range of each SubReg
    Returns:
    label_box -- the tensor label for angluar box regression
    boxes -- the boxes for post process
    """
    label = [np.zeros((train_output_sizes[i], 1, microreg_number,
                       5)) for i in range(1)]
    bboxes_xywh = [np.zeros((max_bbox_per_scale, 4)) for _ in range(1)]
    bbox_count = np.zeros((3,))
    for index, bbox in enumerate(bboxes):
        bbox_coor = bbox[:4]
        if bbox_coor[2] * bbox_coor[3] == 0:
            continue
        bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
        bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]
        best_detect = 0
        xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)
        grid_r = label[best_detect].shape[0]
        grid_c = label[best_detect].shape[1]
        xind = max(0, xind)
        yind = max(0, yind)
        xind = min(xind, grid_r-1)
        yind = min(yind, grid_c-1)

        stride = 8
        x_center = (bbox_coor[0]+bbox_coor[2])/2.0
        x_res = x_center - stride*xind
        best_microreg = int(x_res/3)

        label[best_detect][xind, yind, best_microreg, :] = 0
        label[best_detect][xind, yind, best_microreg, 0:4] = bbox_xywh
        label[best_detect][xind, yind, best_microreg, 4:5] = 1.0

        bbox_ind = int(bbox_count[best_detect] % max_bbox_per_scale)
        bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
        bbox_count[best_detect] += 1

    label_box = label[0]
    boxes = bboxes_xywh[0]

    return label_box, boxes


def boxes_iou(optimal_box, candidate_box):
    """
    Implementation of the iou calculation procedure
    Arguments:
    optimal_box -- the predicted box which has been adopted in the results
    candidate_box -- the predicted box which in the backup set
    Returns:
    the IoU ratio between optimal_box and candidate_box
    """
    optimal_box = np.array(optimal_box)
    candidate_box = np.array(candidate_box)

    optimal_boxes_area = (optimal_box[..., 2] - optimal_box[..., 0]) * (optimal_box[..., 3] - optimal_box[..., 1])
    candidate_box_area = (candidate_box[..., 2] - candidate_box[..., 0]) \
                         * (candidate_box[..., 3] - candidate_box[..., 1])
    leftup_coordinate       = np.maximum(optimal_box[..., :2], candidate_box[..., :2])
    rightdown_coordinate    = np.minimum(optimal_box[..., 2:], candidate_box[..., 2:])
    inter_section = np.maximum(rightdown_coordinate - leftup_coordinate, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = optimal_boxes_area + candidate_box_area - inter_area
    iou_value          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return iou_value


def soft_nms(predicted_boxes, sigma=0.3):
    """
    :param predicted_boxes: (xmin, ymin, xmax, ymax, score)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    optimal_bboxes = []
    while len(predicted_boxes) > 0:
        max_ind = np.argmax(predicted_boxes[:, 4])
        max_score_bbox = predicted_boxes[max_ind]
        optimal_bboxes.append(max_score_bbox)
        predicted_boxes = np.concatenate([predicted_boxes[: max_ind], predicted_boxes[max_ind + 1:]])
        iou = boxes_iou(max_score_bbox[np.newaxis, :4], predicted_boxes[:, :4])
        decay_weight = np.exp(-(1.0 * iou ** 2 / sigma))
        predicted_boxes[:, 4] = predicted_boxes[:, 4] * decay_weight
        score_mask = predicted_boxes[:, 4] > 0.
        predicted_boxes = predicted_boxes[score_mask]

    return optimal_bboxes


def postprocess_boxes(predicted_box, input_size, confidence_threshold):
    """
    Implementation of the decode procedure

    Arguments:
    predicted_box -- input tensor of shape (BATCH_SIZE, n_H_prev, n_W_prev, n_C_prev)
    input_size -- integer, specifying the shape of original sample
    confidence_threshold -- The predicted values with confidence lower than the threshold are filtered out
    Returns:
    output -- the true coordinate of angular boxes with confidence scores
    """
    valid_scale = [0, np.inf]
    predicted_box = np.array(predicted_box)

    pridicted_xywh = predicted_box[:, 0:4]
    predicted_confi = predicted_box[:, 4]

    # step 1 transform (x_center, y_center, width, height) to (x_downleft, y_downleft, x_topright, y_topright)
    predicted_coord = np.concatenate([pridicted_xywh[:, :2] - pridicted_xywh[:, 2:] * 0.5,
                                pridicted_xywh[:, :2] + pridicted_xywh[:, 2:] * 0.5], axis=-1)

    max_width, max_height = input_size, input_size
    # step 2  cut the predicted boxes to prevent boundary crossing
    predicted_coord = np.concatenate([np.maximum(predicted_coord[:, :2], [0, 0]),
                                np.minimum(predicted_coord[:, 2:], [max_width - 1, max_height - 1])], axis=-1)
    invalid_mask = np.logical_or((predicted_coord[:, 0] > predicted_coord[:, 2]),
                                 (predicted_coord[:, 1] > predicted_coord[:, 3]))
    predicted_coord[invalid_mask] = 0

    # step 3  Delete the invalid forecast
    bboxes_scale = np.sqrt(np.multiply.reduce(predicted_coord[:, 2:4] - predicted_coord[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # step 4  Delete the predicted boxes below the threshold
    scores = predicted_confi
    score_mask = scores > confidence_threshold
    mask = np.logical_and(scale_mask, score_mask)
    true_coord, scores = predicted_coord[mask], scores[mask]

    return np.concatenate([true_coord, scores[:, np.newaxis]], axis=-1)





