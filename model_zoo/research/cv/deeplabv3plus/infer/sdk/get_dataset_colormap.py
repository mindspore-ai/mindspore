# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""generate the colormap"""
from six.moves import range
import numpy as np


_DATASET_MAX_ENTRY = 512


def bit_get(val, idx):
    return (val >> idx) & 1


def create_pascal_label_colormap():
    colormap = np.zeros((_DATASET_MAX_ENTRY, 3), dtype=int)
    ind = np.arange(_DATASET_MAX_ENTRY, dtype=int)

    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label, Got {}'.format(label.shape))

    if np.max(label) >= _DATASET_MAX_ENTRY:
        raise ValueError('label value too large: {} >= {}'.format(
            np.max(label), _DATASET_MAX_ENTRY))

    colormap = create_pascal_label_colormap()
    return colormap[label]
