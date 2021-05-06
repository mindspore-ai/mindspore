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
'''post process for 310 inference'''
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='fasterrcnn_export')
parser.add_argument("--result_path", type=str, required=True, help="result file path")
parser.add_argument("--label_file", type=str, required=True, help="label file")
args = parser.parse_args()

def read_label(label_file):
    f = open(label_file, "r")
    lines = f.readlines()

    img_label = {}
    for line in lines:
        img_id = line.split(":")[0]
        label = line.split(":")[1]
        img_label[img_id] = label

    return img_label

def cal_acc(result_path, label_file):
    step = 0
    sum_a = 0
    img_label = read_label(label_file)

    files = os.listdir(result_path)
    for file in files:
        full_file_path = os.path.join(result_path, file)
        if os.path.isfile(full_file_path):
            result = np.fromfile(full_file_path, dtype=np.float32).reshape(1, 1000)
            pred = np.argmax(result, axis=1)
            step = step + 1
            if pred == int(img_label[file[:-6]]):
                sum_a = sum_a + 1

    print("========step:{}========".format(step))
    print("========sum:{}========".format(sum_a))
    accuracy = sum_a * 100.0 / step
    print("========accuracy:{}========".format(accuracy))

if __name__ == "__main__":
    cal_acc(args.result_path, args.label_file)
