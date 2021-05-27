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

parser = argparse.ArgumentParser(description='squeeze calcul top1 and top5 acc')
parser.add_argument("--dataset", type=str, required=True, default="imagenet", help="dataset: Imagenet or Cifar10")
parser.add_argument("--result_path", type=str, required=True, default='', help="result file path")
parser.add_argument("--label_file", type=str, required=True, default='', help="label file")
args = parser.parse_args()


def get_top5_acc(top_arg, gt_class):
    sub_count = 0
    for top5, gt in zip(top_arg, gt_class):
        if gt in top5:
            sub_count += 1
    return sub_count


def read_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    img_dict = {}
    for line in lines:
        img_id = line.split(':')[0]
        label = line.split(':')[1]
        img_dict[img_id] = label
    return img_dict


def cal_acc_cifar10(result_path, label_path):
    img_tot = 0
    top1_correct = 0
    top5_correct = 0
    result_shape = (1, 10)
    files = os.listdir(result_path)
    for file in files:
        full_file_path = os.path.join(result_path, file)
        if os.path.isfile(full_file_path):
            result = np.fromfile(full_file_path, dtype=np.float32).reshape(result_shape)
            label_file = os.path.join(label_path, file)
            gt_classes = np.fromfile(label_file, dtype=np.int32)
            top1_output = np.argmax(result, (-1))
            top5_output = np.argsort(result)[:, -5:]
            t1_correct = np.equal(top1_output, gt_classes).sum()
            top1_correct += t1_correct
            top5_correct += get_top5_acc(top5_output, [gt_classes])
            img_tot += 1
    acc1 = 100 * top1_correct / img_tot
    acc5 = 100 * top5_correct / img_tot
    print('total={}, top1_correct={}, acc={:.2f}%'.format(img_tot, top1_correct, acc1))
    print('total={}, top5_correct={}, acc={:.2f}%'.format(img_tot, top5_correct, acc5))


def cal_acc_imagenet(result_path, label_file):
    img_label = read_label(label_file)
    img_tot = 0
    top1_correct = 0
    top5_correct = 0
    result_shape = (1, 1000)
    files = os.listdir(result_path)
    for file in files:
        full_file_path = os.path.join(result_path, file)
        if os.path.isfile(full_file_path):
            result = np.fromfile(full_file_path, dtype=np.float32).reshape(result_shape)
            gt_classes = int(img_label[file.split('.')[0]])

            top1_output = np.argmax(result, (-1))
            top5_output = np.argsort(result)[:, -5:]

            t1_correct = np.equal(top1_output, gt_classes).sum()
            top1_correct += t1_correct
            top5_correct += get_top5_acc(top5_output, [gt_classes])
            img_tot += 1
    acc1 = 100 * top1_correct / img_tot
    acc5 = 100 * top5_correct / img_tot
    print('total={}, top1_correct={}, acc={:.2f}%'.format(img_tot, top1_correct, acc1))
    print('total={}, top5_correct={}, acc={:.2f}%'.format(img_tot, top5_correct, acc5))


if __name__ == '__main__':
    if args.dataset.lower() == "cifar10":
        cal_acc_cifar10(args.result_path, args.label_file)
    else:
        cal_acc_imagenet(args.result_path, args.label_file)
