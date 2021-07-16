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
"""datasets"""
import os
import random
import numpy as np

random.seed(1)
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif', '.tiff']


def is_image_file(filename):
    """if a file is a image"""
    return any(filename.lower().endswith(extension)
               for extension in IMG_EXTENSIONS)


def make_dataset(dir_path, mode, selected_attrs):
    """ make dataset """
    assert mode in ['train', 'val',
                    'test'], "Mode [{}] is not supportable".format(mode)
    assert os.path.isdir(dir_path), '%s is not a valid directory' % dir_path
    lines = [
        line.rstrip() for line in open(
            os.path.join(dir_path, 'anno', 'list_attr_celeba.txt'), 'r')
    ]
    all_attr_names = lines[1].split()
    attr2idx = {}
    idx2attr = {}
    for i, attr_name in enumerate(all_attr_names):
        attr2idx[attr_name] = i
        idx2attr[i] = attr_name

    lines = lines[2:]
    if mode == 'train':
        lines = lines[:-20000]  # train set contains 182599 images
    if mode == 'val':
        lines = lines[-20000:-18800]  # val set contains 200 images
    if mode == 'test':
        lines = lines[-18800:]  # test set contains 18800 images

    items = []
    for i, line in enumerate(lines):
        split = line.split()
        filename = split[0]
        values = split[1:]
        label = []
        for attr_name in selected_attrs:
            idx = attr2idx[attr_name]
            label.append(np.float32(values[idx] == '1'))
        items.append([filename, label])
    return items
