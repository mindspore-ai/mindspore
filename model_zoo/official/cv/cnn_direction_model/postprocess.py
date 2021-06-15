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

parser = argparse.ArgumentParser(description='post process for cnn')
parser.add_argument("--result_path", type=str, required=True, help="result file path")
parser.add_argument("--label_path", type=str, required=True, help="label file")
args = parser.parse_args()

def cal_acc(result_path, label_path):
    img_total = 0
    top1_correct = 0

    result_shape = (1, 2)

    files = os.listdir(result_path)
    for file in files:
        full_file_path = os.path.join(result_path, file)
        if os.path.isfile(full_file_path):
            result = np.fromfile(full_file_path, dtype=np.float32).reshape(result_shape)
            label_file = os.path.join(label_path, file.split(".bin")[0][:-2] + ".bin")
            gt_classes = np.fromfile(label_file, dtype=np.int32)

            top1_output = np.argmax(result, (-1))

            t1_correct = np.equal(top1_output, gt_classes).sum()
            top1_correct += t1_correct
            img_total += 1

    acc1 = 100.0 * top1_correct / img_total
    print('top1_correct={}, total={}, acc={:.2f}%'.format(top1_correct, img_total, acc1))

if __name__ == "__main__":
    cal_acc(args.result_path, args.label_path)
