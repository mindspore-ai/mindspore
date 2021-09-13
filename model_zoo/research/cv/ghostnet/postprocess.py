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
import argparse
import numpy as np

batch_size = 1
parser = argparse.ArgumentParser(description="ghostnet inference")
parser.add_argument("--result_path", type=str, required=True, help="result files path.")
parser.add_argument("--label_path", type=str, required=True, help="image file path.")
args = parser.parse_args()


def get_top5_acc(top5_arg, gt_class):
    """
    get top5 accuracy
    Args:
        top5_arg:
        gt_class:

    Returns:

    """
    sub_count = 0
    for top5, gt in zip(top5_arg, gt_class):
        if gt in top5:
            sub_count += 1
    return sub_count


def cal_acc_imagenet(result_path, label_path):
    """
    top1 accuracy, top5 accuracy
    Args:
        result_path:
        label_path:

    Returns:

    """
    files = os.listdir(result_path)
    with open(label_path, "r") as label:
        labels = json.load(label)
    result_shape = (1, 1000)
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

    cal_acc_imagenet(args.result_path, args.label_path)
