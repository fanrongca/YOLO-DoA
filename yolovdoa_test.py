#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2021. All rights reserved.
#
#   Author      : Rong Fan
#   Created date: 2021-11-18 12:06:21
#   Description : the Main file of Prediction
#
# ================================================================

import tensorflow as tf
import os
import sys
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from yolovdoa_train import GetProcessedData
from yolovdoa_train import _parse_record
from yolovdoa_train import postprocess_predictedbox

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

TRAINING_STEPS = 32
BATCH_SIZE = 200
TRAINING_EPOCH = 1

test_path = r'D:\FEIQ\RongFan\YOLO\Test.tfrecord'
saved_model_dir = r'D:\FEIQ\RongFan\YOLO\Submit'
Test_Dir = saved_model_dir + r'\\Predict\\'


def YOLO_DOA_Predict():
    """
    Implementation of prediction procedure
    Arguments:
    Returns:
    """
    dataset_test = tf.data.TFRecordDataset(test_path)
    dataset_test = dataset_test.map(_parse_record)
    dataset_test = dataset_test.batch(BATCH_SIZE).repeat(TRAINING_EPOCH)
    iterator_test = dataset_test.make_one_shot_iterator()
    data_test = iterator_test.get_next()

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.device('/cpu:0'):
        with tf.Session(config=config) as sess:
            signature_key = 'test_signature'
            tf_input_data = 'tf_input_data'
            tf_label_box = 'tf_label_box'
            tf_true_boxes = 'tf_true_boxes'
            tf_is_training = 'is_training'
            predicted_box = 'predicted_box'

            meta_graph_def = tf.saved_model.loader.load(sess, ['test_saved_model'], saved_model_dir)
            signature = meta_graph_def.signature_def
            tf_input_data = signature[signature_key].inputs[tf_input_data].name
            tf_label_box = signature[signature_key].inputs[tf_label_box].name
            tf_true_boxes = signature[signature_key].inputs[tf_true_boxes].name
            tf_is_training = signature[signature_key].inputs[tf_is_training].name

            predicted_box = signature[signature_key].outputs[predicted_box].name

            tf_input_data = sess.graph.get_tensor_by_name(tf_input_data)
            tf_label_box = sess.graph.get_tensor_by_name(tf_label_box)
            tf_true_boxes = sess.graph.get_tensor_by_name(tf_true_boxes)
            tf_is_training = sess.graph.get_tensor_by_name(tf_is_training)

            predicted_box = sess.graph.get_tensor_by_name(predicted_box)

            if not os.path.exists(Test_Dir):
                os.makedirs(Test_Dir)
            for epoch in range(TRAINING_EPOCH):
                for step in range(TRAINING_STEPS):
                    data_test_batch = sess.run(data_test)
                    input_data, label_box,\
                    true_boxes, labels \
                        = GetProcessedData(data_test_batch, is_test=True)
                    pred_box_value \
                        = sess.run(
                        predicted_box,
                        feed_dict={
                        tf_input_data: input_data,
                        tf_label_box: label_box,
                        tf_true_boxes: true_boxes,
                        tf_is_training: False
                    })

                    valid_csv_name = str(step) + '_valid.csv'
                    true_csv_name = str(step) + '_true.csv'
                    postprocess_predictedbox(Test_Dir, pred_box_value,
                                 valid_csv_name, labels, true_csv_name)

                    percent = 100 * (step + 1) * 1.0/TRAINING_STEPS
                    sys.stdout.write('\r' + "[%.1f%%]" % percent)
                    sys.stdout.flush()
    pass


def read_csv(label_file, predict_file, start_id):
    """
    Implementation of reding predicted angles form csv file
    Arguments:
        label_file --  the csv file contains true incident directions
        predict_file --  the csv file contains predicted boxes
        start_id --  int, global serial number of the incident directions
    Returns:
        reslut --  array contains true incident directions and corresponding predicted angles
    """
    y_true = pd.read_csv(label_file, header=None).values

    y_predict = pd.read_csv(predict_file, header=None).values
    y_pred = np.zeros([y_predict.shape[0], 2])
    y_pred[:, 0] = y_predict[:, 0] + y_predict[:, 2]
    y_pred[:, 1] = y_predict[:, 5] + y_predict[:, 7]
    y_pred = np.sort(y_pred, axis=1)
    y_pred /= 2.0

    reslut = np.zeros(y_true.shape[0] * y_true.shape[1] * 3 + y_true.shape[0] * 2).reshape(y_true.shape[0], -1)
    Id = np.arange(1, y_true.shape[0] + 1)
    Id = Id[:, np.newaxis]
    Id += start_id
    Id.reshape(-1, 1)
    reslut[:, 0] = Id[:, 0]
    reslut[:, 1] = y_true[:, 1] - y_true[:, 0]
    for index in range(y_true.shape[1]):
        reslut[:, index * 3 + 2] = y_true[:, index]
        reslut[:, index * 3 + 3] = y_pred[:, index]
        reslut[:, index * 3 + 4] = y_true[:, index] - y_pred[:, index]
        pass
    pass
    return reslut


def merge_muti_files():
    """
    merge the predicted files and labels into one csv file
    Arguments:
    Returns:
        csv_path --  array contains true incident directions and corresponding predicted angles
    """
    root_dir = Test_Dir
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_name = os.path.join(root, file)
            file_list.append(file_name)
            pass
        pass
    pass
    handled_list = []
    merge_result = []
    for index, item in enumerate(file_list):
        exchange = ""
        temp = file_name = item
        dir_name = root_dir
        start_id = int(index/2) * BATCH_SIZE
        if "valid" in file_name:
            exchange = file_name.replace("valid", "true")
            file_name = file_name.replace("valid", "")
        if "true" in file_name:
            exchange = file_name.replace("true", "valid")
            file_name = file_name.replace("true", "")
        if file_name in handled_list:
            pass
        else:
            handled_list.append(file_name)
            item_label = os.path.join(dir_name, exchange)
            item_predict = os.path.join(dir_name, item)
            if os.path.isfile(item_label) and os.path.isfile(item):
                if "valid" in temp:
                    result = read_csv(item_label, item_predict, start_id)
                    merge_result.append(result)
                    pass
                else:
                    result = read_csv(item_predict, item_label, start_id)
                    merge_result.append(result)
                    pass
                pass
            pass

        pass
    # save the merge_result
    result_array = np.array(merge_result)
    result_array = np.reshape(result_array, (-1, 8))
    csv_path = saved_model_dir + r'\\predict.csv'
    np.savetxt(csv_path, result_array, fmt='%.3f', delimiter=',')
    return csv_path


def accuracy_test(y_real, y_pred):
    """
    calculation of the RMSE between true angles and predicted angles
    Arguments:
        y_real -- true angles
        y_pred -- predicted angles
    Returns:
        acc --  RMSE
    """
    with tf.name_scope('accuracy'):
        acc = np.sqrt(mean_squared_error(y_real,y_pred ))
    return acc


def calculate_rmse_90(csv_path):
    """
    calculation of the RMSE between true angles and predicted angles in the range of [-90,90)
    Arguments:
        csv_path -- the file path of merged files
    Returns:
        test_acc --  RMSE
    """
    angles = pd.read_csv(csv_path, header=None).values
    y_true = angles[:, [2, 5]]
    y_pred = angles[:, [3, 6]]
    test_acc = accuracy_test(y_true, y_pred)
    # print("±90° RMSE:%.2f" % test_acc)
    return test_acc
    pass


def GetIndexs(arrayData, begin, end):
    """
    get the indexs of angles which are out of range [begin, end)
    Arguments:
        arrayData -- the array of angles
        begin -- begin index
        end -- end index
    Returns:
        Indexs --  int,the indexs of angles which are out of range [begin, end)
    """
    Indexs = []
    for index, item in enumerate(arrayData):
        if item <begin or item >end:
            Indexs.append(index)
    Indexs = np.array(Indexs)
    return Indexs
    pass


def calculate_rmse_85(csv_path):
    """
    calculation of the RMSE between true angles and predicted angles in the range of [-85,85)
    Arguments:
        csv_path -- the file path of merged files
    Returns:
        test_acc --  RMSE
    """
    angles = pd.read_csv(csv_path, header=None).values
    angel_1 = angles[:, 2]
    angel_2 = angles[:, 5]
    begin = 5
    end = 175
    out_index_1 = GetIndexs(angel_1, begin, end)
    out_index_2 = GetIndexs(angel_2, begin, end)
    out_indexs = np.concatenate((out_index_1, out_index_2))
    out_indexs = np.unique(out_indexs)
    out_indexs = out_indexs.astype('int32')
    if out_indexs.size != 0:
        y_handel = np.delete(angles, out_indexs, axis=0)

    y_true = y_handel[:, [2, 5]]
    y_pred = y_handel[:, [3, 6]]
    test_acc = accuracy_test(y_true, y_pred)
    # print("±85° RMSE:%.2f" % test_acc)
    return test_acc
    pass


if __name__ == '__main__':
    # Read the tfrecord file and generate the predicted box
    YOLO_DOA_Predict()
    # Merge the forecast files into one file
    csv_path = merge_muti_files()
    # Calculate RMSE in the range of [-90°,90°)
    acc_90 = calculate_rmse_90(csv_path)
    # Calculate RMSE in the range of [-85°,85°)
    acc_85 = calculate_rmse_85(csv_path)
    print("±90° RMSE:%.2f" % acc_90)
    print("±85° RMSE:%.2f" % acc_85)


