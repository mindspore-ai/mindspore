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
"""postprocess for 310 inference"""
import os
import argparse
import numpy as np

from mindspore.nn import Top1CategoricalAccuracy


parser = argparse.ArgumentParser("vgg16 quant postprocess")
parser.add_argument("--result_path", type=str, required=True, help="path to inference results.")
parser.add_argument("--label_path", type=str, required=True, help="path to label.npy.")

args, _ = parser.parse_known_args()

def calculate_acc(result_path, label_path):
    """
    Calculate accuracy of VGG16 inference.

    Args:
        result_path (str): the directory or inference result.
        label_path (str): the path of data label in .npy format.
    """
    top1_acc = Top1CategoricalAccuracy()
    labels = np.load(label_path, allow_pickle=True)
    batch_size = 1
    for idx, _ in enumerate(labels):
        f_name = os.path.join(result_path, "VGG16_data_bs" + str(batch_size) + "_" + str(idx) + "_output_0.bin")
        pred = np.fromfile(f_name, np.float32)
        pred = pred.reshape(batch_size, int(pred.shape[0] / batch_size))
        top1_acc.update(pred, labels[idx])
    print("acc: ", top1_acc.eval())


if __name__ == '__main__':
    calculate_acc(args.result_path, args.label_path)
