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
import numpy as np
from src.model_utils.config import config


config.batch_size = config.batch_size_postprocess

def calcul_acc(labels, preds):
    return sum(1 for x, y in zip(labels, preds) if x == y) / len(labels)


def read_label(label_path):
    label_dict = {}
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        file_name = line.split(':')[0]
        label = line.split(':')[1]
        label_dict[file_name] = label
    return label_dict


def get_result(result_path, label_path):
    files = os.listdir(result_path)
    preds = []
    labels = []
    label_dict = read_label(label_path)
    for file in files:
        file_name = file.split('.')[0]
        label = int(label_dict[file_name])
        labels.append(label)
        output = np.fromfile(os.path.join(result_path, file), dtype=np.float32)
        preds.append(np.argmax(output, axis=0))
    acc = calcul_acc(labels, preds)
    print("total{}, accuracy: {}".format(len(labels), acc))


if __name__ == '__main__':
    get_result(config.result_path, config.label_path)
