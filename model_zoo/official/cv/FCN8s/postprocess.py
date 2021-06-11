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
import cv2
from PIL import Image

parser = argparse.ArgumentParser(description="FasterRcnn inference")
parser.add_argument("--image_list", type=str, required=True, help="result file path.")
parser.add_argument("--result_path", type=str, required=True, help="result file path.")
parser.add_argument("--data_path", type=str, required=True, help="mask file path.")
parser.add_argument("--mask_path", type=str, required=True, help="mask file path.")
args = parser.parse_args()

NUM_CLASSES = 21

def get_img_size(file_name):
    img = Image.open(file_name)
    return img.size

def get_resized_size(org_h, org_w, long_size=512):
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

def cal_acc(image_list, data_path, result_path, mask_path):
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
    with open(image_list) as f:
        img_list = f.readlines()

    for img in img_list:
        img_file = os.path.join(data_path, img.strip() + ".jpg")
        org_width, org_height = get_img_size(img_file)

        resize_h, resize_w = get_resized_size(org_height, org_width)

        result_file = os.path.join(result_path, img.strip() + "_0.bin")
        result = np.fromfile(result_file, dtype=np.float32).reshape(21, 512, 512)
        probs_ = result[:, :resize_h, :resize_w].transpose((1, 2, 0))
        probs_ = cv2.resize(probs_.astype(np.float32), (org_width, org_height))
        result_msk = probs_.argmax(axis=2)

        mask_file = os.path.join(mask_path, img.strip() + ".png")
        mask = np.array(Image.open(mask_file), dtype=np.uint8)

        hist += cal_hist(mask.flatten(), result_msk.flatten(), NUM_CLASSES)

    #print(hist)
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('per-class IoU', iu)
    print('mean IoU', np.nanmean(iu))

if __name__ == '__main__':
    cal_acc(args.image_list, args.data_path, args.result_path, args.mask_path)
