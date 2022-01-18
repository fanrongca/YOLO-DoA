#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : PyCharm
#   File name   : yolovdoa_train.py
#   Author      : Rong Fan
#   Created date: 2021-10-18 12:06:21
#   Description :
#
#================================================================


import datetime
import os
import time
import shutil
import numpy as np
import tensorflow as tf
import utils as utils
from yolovdoa import YOLODOA


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
input_shape = 192
output_shape = 180
TRAINING_STEPS = 3000
BATCH_SIZE = 200
TRAINING_EPOCH = 200

train_num = 600000
valid_num = 42000
test_num = 192600
angel_range = 90
CURRENT_MODEL_NAME = "YOLO-DoA"
train_path = r'D:\FEIQ\RongFan\YOLO\Train.tfrecord'
test_path = r'D:\FEIQ\RongFan\YOLO\Test.tfrecord'
valid_path = r'D:\FEIQ\RongFan\YOLO\Valid.tfrecord'
save_model_path = r'D:\FEIQ\RongFan\YOLO\Submit'
max_box_num_per_image = 2  # the number of angular boxes
signal_num = 2  # the numer of incident directions
pad_num = 12    # the numerber of zero padding
save_num = 5    # the numerber of predicted boxes


adam_beta1 = 0.9
adam_beta2 = 0.999
opt_epsilon = 1.0
initial_learning_rate = 0.01
decay_steps = TRAINING_STEPS
decay_rate = 0.96
staircase = True
max_bbox_per_scale = signal_num
anchor_per_scale = 3


# =========================================================================== #
# =========================================================================== #

# Before using tf.layers.batch_normalization, set the attribute to tf.GraphKeys.UPDATE_OPS
def train_optimizer(optimizer, loss, global_step=None):
    """
    Implementation of the Adam optimizer
    Arguments:
    optimizer -- Adam optimizer
    loss -- the tensor for loss value
    global_step -- the training step in global_step times
    Returns:
    train_step -- the current train_step
    """
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = optimizer.minimize(loss, global_step=global_step)
    return train_step


# Parsing the tfrecord file
def _parse_record(example_proto):
    """
    Implementation for parse of tfrecord file
    Arguments:
    example_proto -- info should be parsed
    Returns:
    parsed_features -- the parsed features which are used to build the input sample and lable
    """
    features = {
        # the input sample
        'IQData': tf.FixedLenFeature((), tf.string),
        # the collection of incident directions
        'Label': tf.FixedLenFeature((), tf.string)}
    parsed_features = tf.parse_single_example(example_proto, features=features)
    return parsed_features


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
    label = [np.zeros((train_output_sizes[i], 1, anchor_per_scale,
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


def GetProcessedData(data_batch, is_test=False):
    """
    Implementation of read data from tfrecord file
    Arguments:
    data_batch -- the iterator for training
    is_test -- the status of training or prediction
    Returns:
    sample_data -- the sample data fed into YOLO-DoA
    label_box -- the label data fed into YOLO-DoA
    true_boxes -- the boxes for post process
    Label_Merge -- the boxes for post process
    """
    Data_String = data_batch['IData']
    Data_Merge = []
    eight_zeros = np.zeros(pad_num)
    for data in Data_String:
        Data = np.fromstring(data, dtype=np.float64)
        Data = np.append(Data, eight_zeros)
        Data_Merge.append(Data)
    Data_Merge = np.array(Data_Merge)
    sample_data = np.reshape(Data_Merge, [BATCH_SIZE, -1, 1, 1])

    if not is_test:
        Label_String = data_batch['Label']
        Label_Merge = []
        for labeldata in Label_String:
            LabelData = np.fromstring(labeldata, dtype=np.float64)
            indexs = np.argsort(LabelData, axis=-1)[::-1][0:signal_num]
            Label_Merge.append(indexs)

        Label_Merge = np.array(Label_Merge, dtype=np.float32)
        Label_Merge = np.reshape(Label_Merge, [BATCH_SIZE, -1])
        Label_Merge = np.sort(Label_Merge, axis=1)
    else:
        Label_String = data_batch['Label']
        Label_Merge = []
        for labeldata in Label_String:
            LabelData = np.fromstring(labeldata, dtype=np.float64)
            Label_Merge.append(LabelData)
        Label_Merge = np.array(Label_Merge, dtype=np.float32)
        Label_Merge = np.reshape(Label_Merge, [BATCH_SIZE, -1])
        Label_Merge = np.sort(Label_Merge, axis=1) + angel_range
    gbboxes_iou = np.zeros([BATCH_SIZE, max_box_num_per_image, 5])
    # xmin,ymin,xmax,ymax
    xmin = np.reshape(Label_Merge[:, 0], [BATCH_SIZE])
    x_max = np.reshape(Label_Merge[:, 1], [BATCH_SIZE])

    half_width = 1
    gbboxes_iou[:, 0, 0] = xmin - half_width  # xmin range[0,180)
    gbboxes_iou[:, 0, 1] = 0
    gbboxes_iou[:, 0, 2] = xmin + half_width
    gbboxes_iou[:, 0, 3] = 1

    gbboxes_iou[:, 1, 0] = x_max - half_width  # xmin range[0,180)
    gbboxes_iou[:, 1, 1] = 0
    gbboxes_iou[:, 1, 2] = x_max + half_width
    gbboxes_iou[:, 1, 3] = 1

    strides = [8]
    strides = np.asarray(strides, dtype=np.int32)
    train_output_sizes = [24]

    label_box = np.zeros((BATCH_SIZE, train_output_sizes[0], 1, 3, 5), dtype=np.float32)
    true_boxes = np.zeros((BATCH_SIZE, max_bbox_per_scale, 4), dtype=np.float32)

    train_output_sizes = np.asarray(train_output_sizes, dtype=np.int32)

    for i in range(BATCH_SIZE):
        label_box[i], true_boxes[i] = preprocess_true_boxes(gbboxes_iou[i], train_output_sizes, strides)

    return sample_data, label_box, true_boxes, Label_Merge


def main(_):
    """
    Implementation for training procedure of YOLO-DoA
    Arguments:
    Returns:
    """
    global_step = tf.train.create_global_step()
    tf_input_data = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, input_shape, 1, 1], name='input_data')
    tf_label_box = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 24, 1, 3, 5], name='label_box')
    tf_true_boxes = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, max_bbox_per_scale, 4], name='boxes')
    trainable = tf.placeholder(dtype=tf.bool, name='training')
    yolo_model = YOLODOA(tf_input_data, trainable)
    giou_loss, conf_loss = yolo_model.compute_loss(
        tf_label_box,
        tf_true_boxes)
    total_loss = giou_loss + conf_loss
    predicted_box = yolo_model.get_predict_boxs()
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate, global_step, decay_steps,
        decay_rate, staircase, name='learning_rate')

    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=adam_beta1,
        beta2=adam_beta2,
        epsilon=opt_epsilon)

    train_op = train_optimizer(optimizer, total_loss, global_step=global_step)

    # Train
    dataset_train = tf.data.TFRecordDataset(train_path)
    dataset_train = dataset_train.map(_parse_record)
    dataset_train = dataset_train.shuffle(buffer_size=train_num).batch(BATCH_SIZE).repeat(TRAINING_EPOCH)
    iterator_train = dataset_train.make_one_shot_iterator()
    data_train = iterator_train.get_next()

    # Valid
    dataset_valid = tf.data.TFRecordDataset(valid_path)
    dataset_valid = dataset_valid.map(_parse_record)
    dataset_valid = dataset_valid.shuffle(buffer_size=valid_num).batch(BATCH_SIZE).repeat(TRAINING_EPOCH)
    iterator_valid = dataset_valid.make_one_shot_iterator()
    data_valid = iterator_valid.get_next()


    # TEST
    dataset_test = tf.data.TFRecordDataset(test_path)
    dataset_test = dataset_test.map(_parse_record)
    dataset_test = dataset_test.shuffle(buffer_size=test_num).batch(BATCH_SIZE).repeat(TRAINING_EPOCH)
    iterator_test = dataset_test.make_one_shot_iterator()
    data_test = iterator_test.get_next()

    # region
    Train_Valid_Dir = r'E:\Datas\DATA6\YOLO\Compare\\'
    if os.path.exists(Train_Valid_Dir):
        shutil.rmtree(Train_Valid_Dir)
    os.makedirs(Train_Valid_Dir)
    Test_Dir = r'E:\Datas\DATA6\YOLO\CompareTest\\'
    if os.path.exists(Test_Dir):
        shutil.rmtree(Test_Dir)
    os.makedirs(Test_Dir)
    # endregion

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.device('/gpu:0'):
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            for epoch in range(TRAINING_EPOCH):
                time_start = time.time()
                for i in range(TRAINING_STEPS):
                    data_train_batch = sess.run(data_train)
                    input_data, label_box, true_boxes, labels = GetProcessedData(data_train_batch)

                    _, step, total_loss_value, giou_loss_value, conf_loss_value \
                        = sess.run([train_op, global_step, total_loss, giou_loss, conf_loss
                                    ], feed_dict={
                                        tf_input_data: input_data,
                                        tf_label_box: label_box,
                                        tf_true_boxes: true_boxes,
                                        trainable: True,
                            })

                    if step % 20 == 19:
                        data_valid_batch = sess.run(data_valid)
                        input_data, label_box, true_boxes, labels = GetProcessedData(data_valid_batch)
                        total_loss_valid, giou_loss_valid, conf_loss_valid, pred_box_value \
                            = sess.run([total_loss,
                                        giou_loss,
                                        conf_loss,
                                        predicted_box,
                                        ], feed_dict={
                                        tf_input_data: input_data,
                                        tf_label_box: label_box,
                                        tf_true_boxes: true_boxes,
                                        trainable: False
                            })

                        valid_csv_name = str(step) + '_valid.csv'
                        true_csv_name = str(step) + '_true.csv'
                        post_process(Train_Valid_Dir, pred_box_value, valid_csv_name, labels, true_csv_name)

                        # prediction
                        data_test_batch = sess.run(data_test)
                        input_data, label_box, true_boxes, labels = GetProcessedData(data_test_batch, is_test=True)
                        pred_box_value\
                            = sess.run(predicted_box, feed_dict={
                                        tf_input_data: input_data,
                                        tf_label_box: label_box,
                                        tf_true_boxes: true_boxes,
                                        trainable: False
                            })

                        valid_csv_name = str(step) + '_valid.csv'
                        true_csv_name = str(step) + '_true.csv'
                        post_process(Test_Dir, pred_box_value, valid_csv_name, labels, true_csv_name)

                        print("%d,%g,%g,%g,%g,%g,%g" % (step, total_loss_value,
                                                        giou_loss_value,
                                                        conf_loss_value,
                                                        total_loss_valid,
                                                        giou_loss_valid,
                                                        conf_loss_valid,))
                time_end = time.time()
                print("%d epoch time=%f" % (epoch+1, time_end - time_start))

                # save the model
                if epoch > 0 and epoch % 2 == 1:
                    floderName = CURRENT_MODEL_NAME + '_'
                    floderName += datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
                    floderName += "_Epoch_" + str(epoch+1)
                    filePath = os.path.join(save_model_path, floderName)
                    builder = tf.saved_model.builder.SavedModelBuilder(filePath)
                    inputs_params = {'tf_input_data': tf.saved_model.utils.build_tensor_info(tf_input_data),
                                     'tf_label_box': tf.saved_model.utils.build_tensor_info(tf_label_box),
                                     'tf_true_boxes': tf.saved_model.utils.build_tensor_info(tf_true_boxes),
                                     'trainable': tf.saved_model.utils.build_tensor_info(trainable)
                                     }
                    outputs = {'predicted_box': tf.saved_model.utils.build_tensor_info(predicted_box)}
                    signature = tf.saved_model.signature_def_utils.build_signature_def(inputs_params, outputs,
                                                                                       'test_sig_name')
                    builder.add_meta_graph_and_variables(sess, ['test_saved_model'], {'test_signature': signature})
                    builder.save()
                    del builder
            pass
        pass
    pass


def post_process(Save_Dir, pred_box_value, csv_name, labels, true_csv_name):
    """
    Implementation for post process of YOLO-DoA
    Arguments:
    Save_Dir -- file dir for saving the predicted boxes during the training
    pred_box_value -- the 72 predicted boxes need to eliminate the redundant boxes
    csv_name -- the file name of predicted boxes
    labels -- the true labels of DoAs
    true_csv_name -- the file name of true angular boxes
    Returns:
    """
    result = []
    for i in range(BATCH_SIZE):
        pred_bbox = np.reshape(pred_box_value[i], (-1, 5))
        bboxes = utils.postprocess_boxes(pred_bbox, output_shape, output_shape, 0.1)
        bboxes = utils.nms(bboxes)
        bboxes = np.asarray(bboxes, dtype=np.float32)
        final_box = np.zeros(save_num * 5)
        save_len = bboxes.shape[0]
        if save_len > save_num:
            save_len = save_num
        for save_index in range(save_len):
            final_box[5*save_index:5*save_index+5] = bboxes[save_index]
        result.append(final_box)
    result = np.reshape(result, [BATCH_SIZE, -1])
    csv_path = Save_Dir + csv_name
    np.savetxt(csv_path, result, fmt='%s', delimiter=',')
    csv_path = Save_Dir + true_csv_name
    np.savetxt(csv_path, labels, fmt='%s', delimiter=',')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.app.run()
