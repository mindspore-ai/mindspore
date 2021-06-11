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
from src.model_utils.config import config as cf

batch_Size = 1


def is_eq(pred_lbl, target):
    pred_diff = len(target) - len(pred_lbl)
    if pred_diff > 0:
        pred_lbl.extend([10] * pred_diff)
    return pred_lbl == target


def get_prediction(y_pred):
    seq_len, batch_size, _ = y_pred.shape
    indices = y_pred.argmax(axis=2)
    lens = [seq_len] * batch_size
    pred_lbl = []
    for i in range(batch_size):
        idx = indices[:, i]
        last_idx = 10
        pred_lbl = []
        for j in range(lens[i]):
            cur_idx = idx[j]
            if cur_idx not in [last_idx, 10]:
                pred_lbl.append(cur_idx)
            last_idx = cur_idx
    return pred_lbl


def calcul_acc(y_pred, y):
    correct_num = 0
    total_num = 0
    for b_idx, target in enumerate(y):
        if is_eq(y_pred[b_idx], target):
            correct_num += 1
        total_num += 1
    if total_num == 0:
        raise RuntimeError('Accuracy can not be calculated, because the number of samples is 0.')
    return correct_num / total_num


def get_result(result_path, label_path):
    files = os.listdir(result_path)
    preds = []
    labels = []
    label_dict = {}
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            label_dict[line.split(',')[0]] = np.array(
                line.replace('\n', '').replace('[', '').replace(']', '').split(',')[1:]).astype(dtype=int).tolist()
    for file in files:
        label = label_dict[file]
        labels.append(label)
        resultPath = os.path.join(result_path, file)
        output = np.fromfile(resultPath, dtype=np.float16).reshape((-1, batch_Size, 11))
        preds.append(get_prediction(output))
    acc = round(calcul_acc(preds, labels), 3)
    print("Total data: {}, accuracy: {}".format(len(labels), acc))


if __name__ == '__main__':
    get_result(cf.result_path, cf.label_path)
