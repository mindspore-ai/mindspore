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
"""common"""
import random
import os
import numpy as np


def get_patch(*args, patch_size=96, scale=2, input_large=False):
    """get_patch"""
    ih, iw = args[0].shape[:2]

    tp = patch_size
    ip = tp // scale

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    if not input_large:
        tx, ty = scale * ix, scale * iy
    else:
        tx, ty = ix, iy

    ret = [args[0][iy:iy + ip, ix:ix + ip, :], *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]]

    return ret


def set_channel(*args, n_channels=3):
    """set_channel"""
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img[:, :, :n_channels]

    return [_set_channel(a) for a in args]


def np2Tensor(*args, rgb_range=255):
    """ np2Tensor"""
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        input_data = np_transpose.astype(np.float32)
        output = input_data * (rgb_range / 255)
        return output
    return [_np2Tensor(a) for a in args]


def augment(*args, hflip=True, rot=True):
    """augment("""
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        """augment"""
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(a) for a in args]


def search(root, target="JPEG"):
    """search"""
    item_list = []
    items = os.listdir(root)
    for item in items:
        path = os.path.join(root, item)
        if os.path.isdir(path):
            item_list.extend(search(path, target))
        elif path.split('/')[-1].startswith(target):
            item_list.append(path)
        elif target in (path.split('/')[-2], path.split('/')[-3], path.split('/')[-4]):
            item_list.append(path)
    return item_list
