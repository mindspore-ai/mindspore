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
"""generate data and label needed for AIR model inference"""
import os
import sys
import shutil
import cv2
import numpy as np
from PIL import Image


prefix = "deeplabv3_data_bs_"
data_path = "./data"
if os.path.exists(data_path):
    shutil.rmtree(data_path)
os.makedirs(data_path)


def get_img_size(file_name):
    img = Image.open(file_name)
    return img.size


def resize_long(img, long_size=513):
    h, w, _ = img.shape
    if h > w:
        new_h = long_size
        new_w = int(1.0 * long_size * w / h)
    else:
        new_w = long_size
        new_h = int(1.0 * long_size * h / w)
    imo = cv2.resize(img, (new_w, new_h))
    return imo


def pre_process(args, img_, crop_size=513):
    # resize
    img_ = resize_long(img_, crop_size)
    resize_h, resize_w, _ = img_.shape

    # mean, std
    image_mean = np.array(args.image_mean)
    image_std = np.array(args.image_std)
    img_ = (img_ - image_mean) / image_std

    # pad to crop_size
    pad_h = crop_size - img_.shape[0]
    pad_w = crop_size - img_.shape[1]
    if pad_h > 0 or pad_w > 0:
        img_ = cv2.copyMakeBorder(img_, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    # hwc to chw
    img_ = img_.transpose((2, 0, 1))
    return img_, resize_h, resize_w


def eval_batch(args, img_lst, crop_size, index):
    batch_size = len(img_lst)
    batch_img = np.zeros((batch_size, 3, crop_size, crop_size), dtype=np.float32)
    resize_hw = []
    for l in range(batch_size):
        img_ = img_lst[l]
        img_, resize_h, resize_w = pre_process(args, img_, crop_size)
        batch_img[l] = img_
        resize_hw.append([resize_h, resize_w])

    batch_img = np.ascontiguousarray(batch_img)
    data_dir = os.path.join(data_path, "00_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_file = os.path.join(data_dir, prefix + str(batch_size) + "_" + str(index) + ".bin")
    batch_img.tofile(data_file)

def eval_batch_scales(args, img_lst, scales, base_crop_size, index):
    sizes_ = [int((base_crop_size - 1) * sc) + 1 for sc in scales]
    return eval_batch(args, img_lst, crop_size=sizes_[0], index=index)


def generate_data():
    """
    Generate data and label needed for AIR model inference at Ascend310 platform.
    """
    config.scales = config.scales_list[config.scales_type]
    args = config
    # data list
    with open(args.data_lst) as f:
        img_lst = f.readlines()

    # evaluate
    batch_img_lst = []
    batch_msk_lst = []
    shape_lst = []
    for i, line in enumerate(img_lst):
        ori_img_path, ori_msk_path = line.strip().split(" ")
        img_path = "VOCdevkit" + ori_img_path.split("VOCdevkit")[1]
        msk_path = "VOCdevkit" + ori_msk_path.split("VOCdevkit")[1]
        img_path = os.path.join(args.data_root, img_path)
        msk_path = os.path.join(args.data_root, msk_path)
        org_width, org_height = get_img_size(img_path)
        shape_lst.append([org_height, org_width])
        img_ = cv2.imread(img_path)
        msk_ = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        batch_img_lst.append(img_)
        batch_msk_lst.append(msk_)
        eval_batch_scales(args, batch_img_lst, scales=args.scales, base_crop_size=args.crop_size, index=i)
        label_dir = os.path.join(data_path, "01_label")
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        label_path = os.path.join(label_dir, prefix + str(len(batch_img_lst)) + "_" + str(i) + ".bin")
        msk_.tofile(label_path)
        batch_img_lst = []
        batch_msk_lst = []
    np.save(os.path.join(data_path, "shape.npy"), shape_lst)


if __name__ == "__main__":
    sys.path.append("..")
    from model_utils.config import config

    generate_data()
