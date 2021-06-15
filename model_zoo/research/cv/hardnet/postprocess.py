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
import json
import numpy as np
from mindspore.nn import Top1CategoricalAccuracy, Top5CategoricalAccuracy
from src.config import config as cfg
parser = argparse.ArgumentParser(description="postprocess")
label_path = "./preprocess_Result/cifar10_label_ids.npy"
parser.add_argument("--result_dir", type=str, default="./result_Files", help="result files path.")
parser.add_argument('--dataset_name', type=str, choices=["cifar10", "imagenet2012"], default="cifar10")
parser.add_argument("--label_dir", type=str, default=label_path, help="image file path.")
parser.add_argument("--config_path", type=str, default="../default_config.yaml", help="config file path.")
args = parser.parse_args()

cfg.config_path = args.config_path
def calcul_acc(lab, preds):
    return sum(1 for x, y in zip(lab, preds) if x == y) / len(lab)
if __name__ == '__main__':
    batch_size = 1
    top1_acc = Top1CategoricalAccuracy()
    rst_path = args.result_dir
    label_list = []
    pred_list = []
    file_list = os.listdir(rst_path)
    top5_acc = Top5CategoricalAccuracy()
    with open('./preprocess_Result/imagenet_label.json', "r") as label:
        labels = json.load(label)
    for f in file_list:
        label = f.split("_0.bin")[0] + ".JPEG"
        label_list.append(labels[label])
        pred = np.fromfile(os.path.join(rst_path, f), np.float32)
        pred = pred.reshape(batch_size, int(pred.shape[0] / batch_size))
        top1_acc.update(pred, [labels[label],])
        top5_acc.update(pred, [labels[label],])
    print("Top1 acc: ", top1_acc.eval())
    print("Top5 acc: ", top5_acc.eval())
