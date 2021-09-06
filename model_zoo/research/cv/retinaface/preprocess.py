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
"""preprocess"""
from __future__ import print_function
import argparse
import os
import numpy as np
import cv2

from src.config import cfg_res50

cfg = cfg_res50


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process file')
    parser.add_argument('--val_dataset_folder', type=str, default='/home/dataset/widerface/val',
                        help='val dataset folder.')
    args_opt = parser.parse_args()

    # testing dataset
    testset_folder = args_opt.val_dataset_folder
    testset_label_path = os.path.join(args_opt.val_dataset_folder, "label.txt")
    with open(testset_label_path, 'r') as f:
        _test_dataset = f.readlines()
        test_dataset = []
        for im_path in _test_dataset:
            if im_path.startswith('# '):
                test_dataset.append(im_path[2:-1])  # delete '# ...\n'

    # transform data to bin_file
    print('Transform starting')
    img_path = "./bin_file"
    os.makedirs(img_path)
    h_max, w_max = 5568, 1056
    for i, img_name in enumerate(test_dataset):
        image_path = os.path.join(testset_folder, 'images', img_name)

        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)

        # testing scale
        if cfg['val_origin_size']:
            resize = 1
            assert img.shape[0] <= h_max and img.shape[1] <= w_max
            image_t = np.empty((h_max, w_max, 3), dtype=img.dtype)
            image_t[:, :] = (104.0, 117.0, 123.0)
            image_t[0:img.shape[0], 0:img.shape[1]] = img
            img = image_t
        else:
            im_size_min = np.min(img.shape[0:2])
            im_size_max = np.max(img.shape[0:2])
            resize = float(target_size) / float(im_size_min)
            # prevent bigger axis from being more than max_size:
            if np.round(resize * im_size_max) > max_size:
                resize = float(max_size) / float(im_size_max)

            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

            assert img.shape[0] <= max_size and img.shape[1] <= max_size
            image_t = np.empty((max_size, max_size, 3), dtype=img.dtype)
            image_t[:, :] = (104.0, 117.0, 123.0)
            image_t[0:img.shape[0], 0:img.shape[1]] = img
            img = image_t

        scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]], dtype=img.dtype)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)  # [1, c, h, w] (1, 3, 2176, 2176)
        # save bin file
        file_name = "widerface_test" + "_" + str(i) + ".bin"
        file_path = os.path.join(img_path, file_name)
        img.tofile(file_path)
        if i % 50 == 0:
            print("Finish {} files".format(i))
    print("=" * 20, "export bin files finished", "=" * 20)
