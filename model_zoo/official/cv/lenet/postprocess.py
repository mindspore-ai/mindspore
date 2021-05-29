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

batch_size = 1


def calcul_acc(labels, preds):
    return sum(1 for x, y in zip(labels, preds) if x == y) / len(labels)


def get_result(result_path, img_path):
    files = os.listdir(img_path)
    preds = []
    labels = []
    for file in files:
        file_name = file.split('.')[0]
        label = int(file_name.split('_')[-1])
        labels.append(label)
        output = np.fromfile(os.path.join(result_path, file_name + '.bin'), dtype=np.float32)
        preds.append(np.argmax(output, axis=0))
    acc = calcul_acc(labels, preds)
    print("accuracy: {}".format(acc))


if __name__ == '__main__':
    get_result(config.result_path, config.img_path)
