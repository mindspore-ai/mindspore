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
parser.add_argument('--data_root', type=str, default='', help='root path of val data')
parser.add_argument('--data_lst', type=str, default='', help='list of val data')
parser.add_argument('--crop_size', type=int, default=513, help='crop size')
parser.add_argument('--num_classes', type=int, default=21, help='number of classes')
parser.add_argument('--result_path', type=str, default='./result_Files', help='result Files path')
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

def acc_cal():
    hist = np.zeros((args.num_classes, args.num_classes))
    with open(args.data_lst) as f:
        img_lst = f.readlines()

    for line in enumerate(img_lst):
        img_path, msk_path = line[1].strip().split(' ')
        img_file_path = os.path.join(args.data_root, img_path)
        org_width, org_height = get_img_size(img_file_path)
        resize_h, resize_w = get_resized_size(org_height, org_width)

        result_file = os.path.join(args.result_path, os.path.basename(img_path).split('.jpg')[0] + '_0.bin')
        net_out = np.fromfile(result_file, np.float32).reshape(args.num_classes, args.crop_size, args.crop_size)
        probs_ = net_out[:, :resize_h, :resize_w].transpose((1, 2, 0))
        probs_ = cv2.resize(probs_, (org_width, org_height))

        result_msk = probs_.argmax(axis=2)

        msk_path = os.path.join(args.data_root, msk_path)
        mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)

        hist += cal_hist(mask.flatten(), result_msk.flatten(), args.num_classes)

    print(hist)
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('per-class IoU', iu)
    print('mean IoU', np.nanmean(iu))

if __name__ == '__main__':
    acc_cal()
