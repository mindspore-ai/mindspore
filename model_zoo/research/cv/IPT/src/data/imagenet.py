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
"""imagent"""
import os
import random
import io
from PIL import Image

import numpy as np
import imageio

def search(root, target="JPEG"):
    """imagent"""
    item_list = []
    items = os.listdir(root)
    for item in items:
        path = os.path.join(root, item)
        if os.path.isdir(path):
            item_list.extend(search(path, target))
        elif path.split('.')[-1] == target:
            item_list.append(path)
        elif path.split('/')[-1].startswith(target):
            item_list.append(path)
    return item_list


def get_patch_img(img, patch_size=96, scale=2):
    """imagent"""
    ih, iw = img.shape[:2]
    tp = scale * patch_size
    if (iw - tp) > -1 and (ih-tp) > 1:
        ix = random.randrange(0, iw-tp+1)
        iy = random.randrange(0, ih-tp+1)
        hr = img[iy:iy+tp, ix:ix+tp, :3]
    elif (iw - tp) > -1 and (ih - tp) <= -1:
        ix = random.randrange(0, iw-tp+1)
        hr = img[:, ix:ix+tp, :3]
        pil_img = Image.fromarray(hr).resize((tp, tp), Image.BILINEAR)
        hr = np.array(pil_img)
    elif (iw - tp) <= -1 and (ih - tp) > -1:
        iy = random.randrange(0, ih-tp+1)
        hr = img[iy:iy+tp, :, :3]
        pil_img = Image.fromarray(hr).resize((tp, tp), Image.BILINEAR)
        hr = np.array(pil_img)
    else:
        pil_img = Image.fromarray(img).resize((tp, tp), Image.BILINEAR)
        hr = np.array(pil_img)
    return hr


class ImgData():
    """imagent"""
    def __init__(self, args, train=True):
        self.input_large = (args.model == 'VDSR')
        self.scale = args.scale
        self.idx_scale = 0
        self.dataroot = args.dir_data
        self.img_list = search(os.path.join(self.dataroot, "train"), "JPEG")
        self.img_list.extend(search(os.path.join(self.dataroot, "val"), "JPEG"))
        self.img_list = sorted(self.img_list)
        self.train = train
        self.args = args
        self.len = len(self.img_list)
        print("data length:", len(self.img_list))
        if self.args.derain:
            self.derain_dataroot = os.path.join(self.dataroot, "RainTrainL")
            self.derain_img_list = search(self.derain_dataroot, "rainstreak")

    def __len__(self):
        return len(self.img_list)

    def _get_index(self, idx):
        return idx % len(self.img_list)

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_lr = self.img_list[idx]
        lr = imageio.imread(f_lr)
        if len(lr.shape) == 2:
            lr = np.dstack([lr, lr, lr])
        return lr, f_lr

    def _np2Tensor(self, img, rgb_range):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = np_transpose.astype(np.float32)
        tensor = tensor * (rgb_range / 255)
        return tensor

    def __getitem__(self, idx):
        if self.args.model == 'vtip' and self.train and self.args.alltask:
            lr, filename = self._load_file(idx % self.len)
            pair_list = []
            rain = self._load_rain()
            rain = np.expand_dims(rain, axis=2)
            rain = self.get_patch(rain, 1)
            rain = self._np2Tensor(rain, rgb_range=self.args.rgb_range)
            for idx_scale in range(4):
                self.idx_scale = idx_scale
                pair = self.get_patch(lr)
                pair_t = self._np2Tensor(pair, rgb_range=self.args.rgb_range)
                pair_list.append(pair_t)
            return pair_list[3], rain, pair_list[0], pair_list[1], pair_list[2], [self.scale], [filename]
        if self.args.model == 'vtip' and self.train and len(self.scale) > 1:
            lr, filename = self._load_file(idx % self.len)
            pair_list = []
            for idx_scale in range(3):
                self.idx_scale = idx_scale
                pair = self.get_patch(lr)
                pair_t = self._np2Tensor(pair, rgb_range=self.args.rgb_range)
                pair_list.append(pair_t)
            return pair_list[0], pair_list[1], pair_list[2], filename
        if self.args.model == 'vtip' and self.args.derain and self.scale[self.idx_scale] == 1:
            lr, filename = self._load_file(idx % self.len)
            rain = self._load_rain()
            rain = np.expand_dims(rain, axis=2)
            rain = self.get_patch(rain, 1)
            rain = self._np2Tensor(rain, rgb_range=self.args.rgb_range)
            pair = self.get_patch(lr)
            pair_t = self._np2Tensor(pair, rgb_range=self.args.rgb_range)
            return pair_t, rain, filename
        if self.args.jpeg:
            hr, filename = self._load_file(idx % self.len)
            buffer = io.BytesIO()
            width, height = hr.size
            patch_size = self.scale[self.idx_scale]*self.args.patch_size
            if width < patch_size:
                hr = hr.resize((patch_size, height), Image.ANTIALIAS)
                width, height = hr.size
            if height < patch_size:
                hr = hr.resize((width, patch_size), Image.ANTIALIAS)
            hr.save(buffer, format='jpeg', quality=25)
            lr = Image.open(buffer)
            lr = np.array(lr).astype(np.float32)
            hr = np.array(hr).astype(np.float32)
            lr = self.get_patch(lr)
            hr = self.get_patch(hr)
            lr = self._np2Tensor(lr, rgb_range=self.args.rgb_range)
            hr = self._np2Tensor(hr, rgb_range=self.args.rgb_range)
            return lr, hr, filename
        lr, filename = self._load_file(idx % self.len)
        pair = self.get_patch(lr)
        pair_t = self._np2Tensor(pair, rgb_range=self.args.rgb_range)
        return pair_t, filename

    def _load_rain(self):
        idx = random.randint(0, len(self.derain_img_list) - 1)
        f_lr = self.derain_img_list[idx]
        lr = imageio.imread(f_lr)
        return lr

    def get_patch(self, lr, scale=0):
        if scale == 0:
            scale = self.scale[self.idx_scale]
        lr = get_patch_img(lr, patch_size=self.args.patch_size, scale=scale)
        return lr

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
