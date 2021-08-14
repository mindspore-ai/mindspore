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
from PIL import Image
import cv2


parser = argparse.ArgumentParser(description="deeplabv3 accuracy calculation")
parser.add_argument('--crop_size', type=int, default=513, help='crop size')
parser.add_argument('--num_classes', type=int, default=21, help='number of classes')
parser.add_argument('--result_path', type=str, default='./result', help='result Files path')
parser.add_argument('--label_path', type=str, default='./01_label', help='result Files path')
parser.add_argument('--shape_path', type=str, default='./shape.npy', help='path of image shape')
args, _ = parser.parse_known_args()


def get_img_size(file_name):
    img = Image.open(file_name)
    return img.size


def get_resized_size(org_h, org_w, long_size=513):
    if org_h > org_w:
        new_h = long_size
        new_w = int(1.0 * long_size * org_w / org_h)
    else:
        new_w = long_size
        new_h = int(1.0 * long_size * org_h / org_w)
    return new_h, new_w


def cal_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(np.int32) + b[k], minlength=n ** 2).reshape(n, n)


def acc_cal(result_path, label_path, shape_path):
    hist = np.zeros((args.num_classes, args.num_classes))
    mask_shape = np.load(shape_path)
    prefix = "deeplabv3_data_bs_1_"
    for i in range(len(mask_shape)):
        output = os.path.join(result_path, prefix + str(i) + "_output_0.bin")
        net_out = np.fromfile(output, np.float32).reshape(args.num_classes, args.crop_size, args.crop_size)
        ori_height, ori_width = mask_shape[i][0], mask_shape[i][1]
        resize_h, resize_w = get_resized_size(ori_height, ori_width)
        probs_ = net_out[:, :resize_h, :resize_w].transpose((1, 2, 0))
        probs_ = cv2.resize(probs_, (ori_width, ori_height))

        result_msk = probs_.argmax(axis=2)
        label = os.path.join(label_path, prefix + str(i) + ".bin")
        mask = np.fromfile(label, np.uint8).reshape(mask_shape[i])

        hist += cal_hist(mask.flatten(), result_msk.flatten(), args.num_classes)

    print(hist)
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('per-class IoU', iu)
    print('mean IoU', np.nanmean(iu))


if __name__ == '__main__':
    acc_cal(args.result_path, args.label_path, args.shape_path)
