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
import json
import argparse
import numpy as np
from mindspore.nn import Top1CategoricalAccuracy, Top5CategoricalAccuracy


parser = argparse.ArgumentParser(description="postprocess")
parser.add_argument("--result_dir", type=str, required=True, help="result files path.")
parser.add_argument("--label_dir", type=str, required=True, help="image file path.")
parser.add_argument('--dataset_name', type=str, choices=["cifar10", "imagenet2012"], default="cifar10")
args = parser.parse_args()

if __name__ == '__main__':
    top1_acc = Top1CategoricalAccuracy()
    rst_path = args.result_dir
    if args.dataset_name == "cifar10":
        from src.config import cifar_cfg as cfg
        labels = np.load(args.label_dir, allow_pickle=True)
        for idx, label in enumerate(labels):
            f_name = os.path.join(rst_path, "VGG19_data_bs" + str(cfg.batch_size) + "_" + str(idx) + "_0.bin")
            pred = np.fromfile(f_name, np.float32)
            pred = pred.reshape(cfg.batch_size, int(pred.shape[0] / cfg.batch_size))
            top1_acc.update(pred, labels[idx])
        print("acc: ", top1_acc.eval())
    else:
        from src.config import imagenet_cfg as cfg
        top5_acc = Top5CategoricalAccuracy()
        file_list = os.listdir(rst_path)
        with open(args.label_dir, "r") as label:
            labels = json.load(label)
        for f in file_list:
            label = f.split("_0.bin")[0] + ".JPEG"
            pred = np.fromfile(os.path.join(rst_path, f), np.float32)
            pred = pred.reshape(cfg.batch_size, int(pred.shape[0] / cfg.batch_size))
            top1_acc.update(pred, [labels[label],])
            top5_acc.update(pred, [labels[label],])
        print("Top1 acc: ", top1_acc.eval())
        print("Top5 acc: ", top5_acc.eval())
