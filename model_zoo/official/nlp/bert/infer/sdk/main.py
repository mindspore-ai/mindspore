# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
sample script of CLUE infer using SDK run in docker
"""

import argparse
import glob
import os
import time

import MxpiDataType_pb2 as MxpiDataType
import numpy as np
from StreamManagerApi import StreamManagerApi, MxDataInput, InProtobufVector, \
    MxProtobufIn, StringVector

TP = 0
FP = 0
FN = 0


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="bert process")
    parser.add_argument("--pipeline", type=str, default="", help="SDK infer pipeline")
    parser.add_argument("--data_dir", type=str, default="",
                        help="Dataset contain input_ids, input_mask, segment_ids, label_ids")
    parser.add_argument("--label_file", type=str, default="", help="label ids to name")
    parser.add_argument("--output_file", type=str, default="", help="save result to file")
    parser.add_argument("--f1_method", type=str, default="BF1", help="calc F1 use the number label,(BF1, MF1)")
    parser.add_argument("--do_eval", type=bool, default=False, help="eval the accuracy of model ")
    args_opt = parser.parse_args()
    return args_opt


def send_source_data(appsrc_id, filename, stream_name, stream_manager):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.

    Returns:
        bool: send data success or not
    """
    tensor = np.fromfile(filename, dtype=np.int32)
    tensor = np.expand_dims(tensor, 0)
    tensor_package_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package = tensor_package_list.tensorPackageVec.add()
    array_bytes = tensor.tobytes()
    data_input = MxDataInput()
    data_input.data = array_bytes
    tensor_vec = tensor_package.tensorVec.add()
    tensor_vec.deviceId = 0
    tensor_vec.memType = 0
    for i in tensor.shape:
        tensor_vec.tensorShape.append(i)
    tensor_vec.dataStr = data_input.data
    tensor_vec.tensorDataSize = len(array_bytes)

    key = "appsrc{}".format(appsrc_id).encode('utf-8')
    protobuf_vec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b'MxTools.MxpiTensorPackageList'
    protobuf.protobuf = tensor_package_list.SerializeToString()
    protobuf_vec.push_back(protobuf)

    ret = stream_manager.SendProtobuf(stream_name, appsrc_id, protobuf_vec)
    if ret < 0:
        print("Failed to send data to stream.")
        return False
    return True


def send_appsrc_data(args, file_name, stream_name, stream_manager):
    """
    send three stream to infer model, include input ids, input mask and token type_id.

    Returns:
        bool: send data success or not
    """
    input_ids = os.path.realpath(os.path.join(args.data_dir, "00_data", file_name))
    if not send_source_data(0, input_ids, stream_name, stream_manager):
        return False
    input_mask = os.path.realpath(os.path.join(args.data_dir, "01_data", file_name))
    if not send_source_data(1, input_mask, stream_name, stream_manager):
        return False
    token_type_id = os.path.realpath(os.path.join(args.data_dir, "02_data", file_name))
    if not send_source_data(2, token_type_id, stream_name, stream_manager):
        return False
    return True


def read_label_file(label_file):
    """
    Args:
        label_file:
        "address"
        "book"
        ...
    Returns:
        label list
    """
    label_list = [line.strip() for line in open(label_file).readlines()]
    return label_list


def process_infer_to_cluner(args, logit_id, each_label_length=4):
    """
    find label and position from the logit_id tensor.

    Args:
        args: param of config.
        logit_id: shape is [128], example: [0..32.34..0].
        each_label_length: each label have 4 prefix, ["S_", "B_", "M_", "E_"].

    Returns:
        dict of visualization result, as 'position': [9, 10]
    """
    label_list = read_label_file(os.path.realpath(args.label_file))
    find_cluner = False
    result_list = []
    for i, value in enumerate(logit_id):
        if value > 0:
            if not find_cluner:
                start = i
                cluner_name = label_list[(value - 1) // each_label_length]
                find_cluner = True
            else:
                if label_list[(value - 1) // each_label_length] != cluner_name:
                    item = {}
                    item[cluner_name] = [start - 1, i - 2]
                    result_list.append(item)
                    start = i
                    cluner_name = label_list[(value - 1) // each_label_length]
        else:
            if find_cluner:
                item = {}
                item[cluner_name] = [start - 1, i - 2]
                result_list.append(item)
                find_cluner = False

    return result_list


def count_pred_result(args, file_name, logit_id, class_num=41, max_seq_length=128):
    """
    support two method to calc f1 sore, if dataset has two class, suggest using BF1,
    else more than two class, suggest using MF1.
    Args:
        args: param of config.
        file_name: label file name.
        logit_id: output tensor of infer.
        class_num: cluner data default is 41.
        max_seq_length: sentence input length default is 128.

    global:
        TP: pred == target
        FP: in pred but not in target
        FN: in target but not in pred
    """
    label_file = os.path.realpath(os.path.join(args.data_dir, "03_data", file_name))
    label_ids = np.fromfile(label_file, np.int32)
    label_ids.reshape(max_seq_length, -1)
    global TP, FP, FN
    if args.f1_method == "BF1":
        pos_eva = np.isin(logit_id, [i for i in range(1, class_num)])
        pos_label = np.isin(label_ids, [i for i in range(1, class_num)])
        TP += np.sum(pos_eva & pos_label)
        FP += np.sum(pos_eva & (~pos_label))
        FN += np.sum((~pos_eva) & pos_label)
    else:
        target = np.zeros((len(label_ids), class_num), dtype=np.int32)
        pred = np.zeros((len(logit_id), class_num), dtype=np.int32)
        for i, label in enumerate(label_ids):
            if label > 0:
                target[i][label] = 1
        for i, label in enumerate(logit_id):
            if label > 0:
                pred[i][label] = 1
        target = target.reshape(class_num, -1)
        pred = pred.reshape(class_num, -1)
        for i in range(0, class_num):
            for j in range(0, max_seq_length):
                if pred[i][j] == 1:
                    if target[i][j] == 1:
                        TP += 1
                    else:
                        FP += 1
                if target[i][j] == 1 and pred[i][j] != 1:
                    FN += 1


def post_process(args, file_name, infer_result, max_seq_length=128):
    """
    process the result of infer tensor to Visualization results.
    Args:
        args: param of config.
        file_name: label file name.
        infer_result: get logit from infer result
        max_seq_length: sentence input length default is 128.
    """
    # print the infer result
    print("==============================================================")
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result[0].messageBuf)
    res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')
    res = res.reshape(max_seq_length, -1)
    print("output tensor is: ", res.shape)

    logit_id = np.argmax(res, axis=-1)
    logit_id = np.reshape(logit_id, -1)
    cluner_list = process_infer_to_cluner(args, logit_id)
    print(cluner_list)
    with open(args.output_file, "a") as file:
        file.write("{}: {}\n".format(file_name, str(cluner_list)))

    if args.do_eval:
        count_pred_result(args, file_name, logit_id)


def run():
    """
    read pipeline and do infer
    """
    args = parse_args()
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        return

    # create streams by pipeline config file
    with open(os.path.realpath(args.pipeline), 'rb') as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        return

    stream_name = b'im_bertbase'
    infer_total_time = 0
    # input_ids file list, every file content a tensor[1,128]
    file_list = glob.glob(os.path.join(os.path.realpath(args.data_dir), "00_data", "*.bin"))
    for input_ids in file_list:
        file_name = input_ids.split('/')[-1]
        if not send_appsrc_data(args, file_name, stream_name, stream_manager_api):
            return
        # Obtain the inference result by specifying streamName and uniqueId.
        key_vec = StringVector()
        key_vec.push_back(b'mxpi_tensorinfer0')
        start_time = time.time()
        infer_result = stream_manager_api.GetProtobuf(stream_name, 0, key_vec)
        infer_total_time += time.time() - start_time
        if infer_result.size() == 0:
            print("inferResult is null")
            return
        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (infer_result[0].errorCode))
            return
        post_process(args, file_name, infer_result)

    if args.do_eval:
        print("==============================================================")
        precision = TP / (TP + FP)
        print("Precision {:.6f} ".format(precision))
        recall = TP / (TP + FN)
        print("Recall {:.6f} ".format(recall))
        print("F1 {:.6f} ".format(2 * precision * recall / (precision + recall)))
        print("==============================================================")
    print("Infer images sum: {}, cost total time: {:.6f} sec.".format(len(file_list), infer_total_time))
    print("The throughput: {:.6f} bin/sec.".format(len(file_list) / infer_total_time))
    # destroy streams
    stream_manager_api.DestroyAllStreams()


if __name__ == '__main__':
    run()
