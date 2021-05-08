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
post process for 310 inference.
"""
import os
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='lenet_quant inference')
    parser.add_argument('--result_path', type=str, default='', help='result files path')
    parser.add_argument('--label_path', type=str, default='', help='label file path')
    args_, _ = parser.parse_known_args()
    return args_

if __name__ == "__main__":
    args = parse_args()
    path = args.result_path
    label_path = args.label_path
    files = os.listdir(path)
    step = 0
    sum_right = 0
    label_dict = {}
    with open(label_path, 'r') as f:
        for line in f.readlines():
            batch_label = line.strip().split(',')
            label_dict[batch_label[0]] = batch_label[1:]
    for file in files:
        full_file_path = os.path.join(path, file)
        if os.path.isfile(full_file_path):
            label_file = file.split('_0.bin')[0] + '.bin'
            label_array = np.array(label_dict[label_file])
            line = np.fromfile(full_file_path, dtype=np.float32)
            batch_size = label_array.shape[0]
            line_comp = line.reshape(batch_size, int(line.shape[0] / batch_size))
        for i in range(0, batch_size):
            pred = np.argmax(line_comp[i], axis=0)
            step += 1
            if pred == label_array[i].astype(np.int64):
                sum_right += 1
    print("=====step:{}=====".format(step))
    print("=====sum_right:{}=====".format(sum_right))
    accuracy = sum_right * 100.0 / step
    print("=====accuracy:{}=====".format(accuracy))
