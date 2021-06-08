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
"""postprocess"""
import os
import json
import argparse
import numpy as np
from mindspore.nn import Top1CategoricalAccuracy, Top5CategoricalAccuracy

parser = argparse.ArgumentParser(description="postprocess")
parser.add_argument("--dataset", type=str, required=True, help="dataset type.")
parser.add_argument("--result_path", type=str, required=True, help="result files path.")
parser.add_argument("--label_path", type=str, required=True, help="image file path.")
args_opt = parser.parse_args()

if args_opt.dataset == "cifar10":
    from src.config import config1 as config
elif args_opt.dataset == "cifar100":
    from src.config import config2 as config
elif args_opt.dataset == 'imagenet2012':
    from src.config import config3 as config
else:
    raise ValueError("dataset is not support.")

def cal_acc_cifar(result_path, label_path):
    '''calculate cifar accuracy'''
    top1_acc = Top1CategoricalAccuracy()
    top5_acc = Top5CategoricalAccuracy()
    result_shape = (config.batch_size, config.class_num)

    file_num = len(os.listdir(result_path))
    label_list = np.load(label_path)
    for i in range(file_num):
        f_name = args_opt.dataset + "_bs" + str(config.batch_size) + "_" + str(i) + "_0.bin"
        full_file_path = os.path.join(result_path, f_name)
        if os.path.isfile(full_file_path):
            result = np.fromfile(full_file_path, dtype=np.float32).reshape(result_shape)
            gt_classes = label_list[i]

            top1_acc.update(result, gt_classes)
            top5_acc.update(result, gt_classes)
    print("top1 acc: ", top1_acc.eval())
    print("top5 acc: ", top5_acc.eval())

def cal_acc_imagenet(result_path, label_path):
    '''calculate imagenet2012 accuracy'''
    batch_size = 1
    files = os.listdir(result_path)
    with open(label_path, "r") as label:
        labels = json.load(label)

    top1 = 0
    top5 = 0
    total_data = len(files)
    for file in files:
        img_ids_name = file.split('_0.')[0]
        data_path = os.path.join(result_path, img_ids_name + "_0.bin")
        result = np.fromfile(data_path, dtype=np.float32).reshape(batch_size, config.class_num)
        for batch in range(batch_size):
            predict = np.argsort(-result[batch], axis=-1)
            if labels[img_ids_name+".JPEG"] == predict[0]:
                top1 += 1
            if labels[img_ids_name+".JPEG"] in predict[:5]:
                top5 += 1
    print(f"Total data: {total_data}, top1 accuracy: {top1/total_data}, top5 accuracy: {top5/total_data}.")


if __name__ == '__main__':
    if args_opt.dataset.lower() == "cifar10" or args_opt.dataset.lower() == "cifar100":
        cal_acc_cifar(args_opt.result_path, args_opt.label_path)
    else:
        cal_acc_imagenet(args_opt.result_path, args_opt.label_path)
