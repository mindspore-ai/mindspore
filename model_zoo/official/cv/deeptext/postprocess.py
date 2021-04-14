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

"""Evaluation for Deeptext"""
import argparse
import os

import numpy as np
from src.config import config
from src.utils import metrics
from PIL import Image
import mmcv

parser = argparse.ArgumentParser(description="Deeptext evaluation")
parser.add_argument("--result_path", type=str, required=True, help="result file path")
parser.add_argument("--label_path", type=str, required=True, help="label path")
parser.add_argument("--img_path", type=str, required=True, help="img path")
args_opt = parser.parse_args()

config.test_batch_size = 1

def get_pred(file, result_path):
    file_name = file.split('.')[0][3:]
    all_bbox_file = os.path.join(result_path, file_name + "_0.bin")
    all_label_file = os.path.join(result_path, file_name + "_1.bin")
    all_mask_file = os.path.join(result_path, file_name + "_2.bin")
    all_bbox = np.fromfile(all_bbox_file, dtype=np.float32).reshape(config.test_batch_size, 1000, 5)
    all_label = np.fromfile(all_label_file, dtype=np.int32).reshape(config.test_batch_size, 1000, 1)
    all_mask = np.fromfile(all_mask_file, dtype=np.bool).reshape(config.test_batch_size, 1000, 1)

    return all_bbox, all_label, all_mask

def get_gt_bboxes_labels(label_file, img_file):
    img_data = np.array(Image.open(img_file))
    img_data, w_scale, h_scale = mmcv.imresize(
        img_data, (config.img_width, config.img_height), return_scale=True)
    scale_factor = np.array(
        [w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
    img_shape = (config.img_height, config.img_width, 1.0)
    img_shape = np.asarray(img_shape, dtype=np.float32)

    file = open(label_file)
    lines = file.readlines()
    boxes = []
    gt_label = []
    for line in lines:
        label_info = line.split(",")
        boxes.append([float(label_info[0]), float(label_info[1]), float(label_info[2]), float(label_info[3])])
        gt_label.append(int(1))

    gt_bboxes = np.array(boxes)
    gt_bboxes = gt_bboxes * scale_factor

    gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1)
    gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1)

    return gt_bboxes, gt_label

def deeptext_eval_test(result_path='', label_path='', img_path=''):
    eval_iter = 0

    print("\n========================================\n")
    print("Processing, please wait a moment.")
    max_num = 32

    pred_data = []
    files = os.listdir(label_path)
    for file in files:
        eval_iter = eval_iter + 1
        img_file = os.path.join(img_path, file.split('gt_')[1].replace("txt", "jpg"))

        label_file = os.path.join(label_path, file)
        gt_bboxes, gt_labels = get_gt_bboxes_labels(label_file, img_file)

        gt_bboxes = np.array(gt_bboxes).astype(np.float32)

        all_bbox, all_label, all_mask = get_pred(file, result_path)
        all_label = all_label + 1


        for j in range(config.test_batch_size):
            all_bbox_squee = np.squeeze(all_bbox[j, :, :])
            all_label_squee = np.squeeze(all_label[j, :, :])
            all_mask_squee = np.squeeze(all_mask[j, :, :])

            all_bboxes_tmp_mask = all_bbox_squee[all_mask_squee, :]
            all_labels_tmp_mask = all_label_squee[all_mask_squee]

            if all_bboxes_tmp_mask.shape[0] > max_num:
                inds = np.argsort(-all_bboxes_tmp_mask[:, -1])
                inds = inds[:max_num]
                all_bboxes_tmp_mask = all_bboxes_tmp_mask[inds]
                all_labels_tmp_mask = all_labels_tmp_mask[inds]

            pred_data.append({"boxes": all_bboxes_tmp_mask,
                              "labels": all_labels_tmp_mask,
                              "gt_bboxes": gt_bboxes,
                              "gt_labels": gt_labels})

    precisions, recalls = metrics(pred_data)
    print("\n========================================\n")
    for i in range(config.num_classes - 1):
        j = i + 1
        f1 = (2 *  precisions[j] * recalls[j]) / (precisions[j] + recalls[j] + 1e-6)
        print("class {} precision is {:.2f}%, recall is {:.2f}%,"
              "F1 is {:.2f}%".format(j, precisions[j] * 100, recalls[j] * 100, f1 * 100))
        if config.use_ambigous_sample:
            break

if __name__ == '__main__':
    deeptext_eval_test(args_opt.result_path, args_opt.label_path, args_opt.img_path)
