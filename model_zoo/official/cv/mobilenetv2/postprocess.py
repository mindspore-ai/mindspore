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
import argparse
import numpy as np

batch_size = 1
parser = argparse.ArgumentParser(description="mobilenetv2 acc calculation")
parser.add_argument("--result_path", type=str, required=True, help="result files path.")
parser.add_argument("--label_path", type=str, required=True, help="label path.")
args = parser.parse_args()


def calcul_acc(labels, preds):
    return sum(1 for x, y in zip(labels, preds) if x == y) / len(labels)


def get_result(result_path, label_path):
    files = os.listdir(result_path)
    preds = []
    labels = []
    label_dict = {}
    with open(label_path, 'w') as f:
        lines = f.readlines()
        for line in lines:
            label_dict[line.split(',')[0]] = line.split(',')[1]
    for file in files:
        file_name = file.split('.')[0]
        label = int(label_dict[file_name + '.JEPG'])
        labels.append(label)
        resultPath = os.path.join(result_path, file)
        output = np.fromfile(resultPath, dtype=np.float32)
        preds.append(np.argmax(output, axis=0))
    acc = calcul_acc(labels, preds)
    print("accuracy: {}".format(acc))


if __name__ == '__main__':
    get_result(args.result_path, args.label_path)
