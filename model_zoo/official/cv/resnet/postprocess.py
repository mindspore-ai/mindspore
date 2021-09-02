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
"""post process for 310 inference"""
import os
import json
import numpy as np
from src.model_utils.config import config


batch_size = 1

def get_top5_acc(top5_arg, gt_class):
    sub_count = 0
    for top5, gt in zip(top5_arg, gt_class):
        if gt in top5:
            sub_count += 1
    return sub_count

def cal_acc_cifar10(result_path, label_path):
    img_tot = 0
    top1_correct = 0
    top5_correct = 0
    img_tot = 0

    result_shape = (1, 10)

    files = os.listdir(result_path)
    for file in files:
        full_file_path = os.path.join(result_path, file)
        if os.path.isfile(full_file_path):
            result = np.fromfile(full_file_path, dtype=np.float32).reshape(result_shape)
            label_file = os.path.join(label_path, file.split(".bin")[0][:-2] + ".bin")
            gt_classes = np.fromfile(label_file, dtype=np.int32)

            top1_output = np.argmax(result, (-1))
            top5_output = np.argsort(result)[:, -5:]

            t1_correct = np.equal(top1_output, gt_classes).sum()
            top1_correct += t1_correct
            top5_correct += get_top5_acc(top5_output, [gt_classes])
            img_tot += 1

    print(f"Total data: {img_tot}, top1 accuracy: {top1_correct / img_tot}, top5 accuracy: {top5_correct / img_tot}.")

def cal_acc_imagenet(result_path, label_path):
    files = os.listdir(result_path)
    with open(label_path, "r") as label:
        labels = json.load(label)
    result_shape = (1, 1001)
    top1 = 0
    top5 = 0
    total_data = len(files)
    for file in files:
        img_ids_name = file.split('_0.')[0]
        data_path = os.path.join(result_path, img_ids_name + "_0.bin")
        result = np.fromfile(data_path, dtype=np.float32).reshape(result_shape)
        for batch in range(batch_size):
            predict = np.argsort(-result[batch], axis=-1)
            if labels[img_ids_name+".JPEG"] == predict[0]:
                top1 += 1
            if labels[img_ids_name+".JPEG"] in predict[:5]:
                top5 += 1
    print(f"Total data: {total_data}, top1 accuracy: {top1/total_data}, top5 accuracy: {top5/total_data}.")


if __name__ == '__main__':
    if config.dataset.lower() == "cifar10":
        cal_acc_cifar10(config.result_path, config.label_path)
    else:
        cal_acc_imagenet(config.result_path, config.label_path)
