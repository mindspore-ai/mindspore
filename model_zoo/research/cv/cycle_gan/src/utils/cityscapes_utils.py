# Copyright 2020 Huawei Technologies Co., Ltd
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
"""cityscape utils."""

import numpy as np
from PIL import Image

# label name and RGB color map.
label2color = {
    'unlabeled': (0, 0, 0),
    'ego vehicle': (0, 0, 0),
    'rectification border': (0, 0, 0),
    'out of roi': (0, 0, 0),
    'static': (0, 0, 0),
    'dynamic': (111, 74, 0),
    'ground': (81, 0, 81),
    'road': (128, 64, 128),
    'sidewalk': (244, 35, 232),
    'parking': (250, 170, 160),
    'rail track': (230, 150, 140),
    'building': (70, 70, 70),
    'wall': (102, 102, 156),
    'fence': (190, 153, 153),
    'guard rail': (180, 165, 180),
    'bridge': (150, 100, 100),
    'tunnel': (150, 120, 90),
    'pole': (153, 153, 153),
    'polegroup': (153, 153, 153),
    'traffic light': (250, 170, 30),
    'traffic sign': (220, 220, 0),
    'vegetation': (107, 142, 35),
    'terrain': (152, 251, 152),
    'sky': (70, 130, 180),
    'person': (220, 20, 60),
    'rider': (255, 0, 0),
    'car': (0, 0, 142),
    'truck': (0, 0, 70),
    'bus': (0, 60, 100),
    'caravan': (0, 0, 90),
    'trailer': (0, 0, 110),
    'train': (0, 80, 100),
    'motorcycle': (0, 0, 230),
    'bicycle': (119, 11, 32),
    'license plate': (0, 0, 142)
}

def fast_hist(a, b, n):
    k = np.where((a >= 0) & (a < n))[0]
    bc = np.bincount(n * a[k].astype(int) + b[k], minlength=n**2)
    if len(bc) != n**2:
        # ignore this example if dimension mismatch
        return 0
    return bc.reshape(n, n)

def get_scores(hist):
    # Mean pixel accuracy
    acc = np.diag(hist).sum() / (hist.sum() + 1e-12)
    # Per class accuracy
    cl_acc = np.diag(hist) / (hist.sum(1) + 1e-12)
    # Per class IoU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-12)
    return acc, np.nanmean(cl_acc), np.nanmean(iu), cl_acc, iu

class CityScapes:
    """CityScapes util class."""
    def __init__(self):
        self.classes = ['road', 'sidewalk', 'building', 'wall', 'fence',
                        'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                        'sky', 'person', 'rider', 'car', 'truck',
                        'bus', 'train', 'motorcycle', 'bicycle', 'unlabeled']
        self.color_list = []
        for name in self.classes:
            self.color_list.append(label2color[name])
        self.class_num = len(self.classes)

    def get_id(self, img_path):
        """Get train id by img"""
        img = np.array(Image.open(img_path).convert("RGB"))
        w, h, _ = img.shape
        img_tile = np.tile(img, (1, 1, self.class_num)).reshape(w, h, self.class_num, 3)
        diff = np.abs(img_tile - self.color_list).sum(axis=-1)
        ids = diff.argmin(axis=-1)
        return ids
