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
"""CTPN anchor generator."""
import numpy as np
class AnchorGenerator():
    """Anchor generator for CTPN."""
    def __init__(self, config):
        """Anchor generator init method."""
        self.base_size = config.anchor_base
        self.num_anchor = config.num_anchors
        self.anchor_height = config.anchor_height
        self.anchor_width = config.anchor_width
        self.size = self.gen_anchor_size()
        self.base_anchors = self.gen_base_anchors()

    def gen_base_anchors(self):
        """Generate a single anchor."""
        base_anchor = np.array([0, 0, self.base_size - 1, self.base_size - 1], np.int32)
        anchors = np.zeros((len(self.size), 4), np.int32)
        index = 0
        for h, w in self.size:
            anchors[index] = self.scale_anchor(base_anchor, h, w)
            index += 1
        return anchors

    def gen_anchor_size(self):
        """Generate a list of anchor size"""
        size = []
        for width in self.anchor_width:
            for height in self.anchor_height:
                size.append((height, width))
        return size

    def scale_anchor(self, anchor, h, w):
        x_ctr = (anchor[0] + anchor[2]) * 0.5
        y_ctr = (anchor[1] + anchor[3]) * 0.5
        scaled_anchor = anchor.copy()
        scaled_anchor[0] = x_ctr - w / 2  # xmin
        scaled_anchor[2] = x_ctr + w / 2  # xmax
        scaled_anchor[1] = y_ctr - h / 2  # ymin
        scaled_anchor[3] = y_ctr + h / 2  # ymax
        return scaled_anchor

    def _meshgrid(self, x, y):
        """Generate grid."""
        xx = np.repeat(x.reshape(1, len(x)), len(y), axis=0).reshape(-1)
        yy = np.repeat(y, len(x))
        return xx, yy

    def grid_anchors(self, featmap_size, stride=16):
        """Generate anchor list."""
        base_anchors = self.base_anchors
        feat_h, feat_w = featmap_size
        shift_x = np.arange(0, feat_w) * stride
        shift_y = np.arange(0, feat_h) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = np.stack([shift_xx, shift_yy, shift_xx, shift_yy], axis=-1)
        shifts = shifts.astype(base_anchors.dtype)
        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.reshape(-1, 4)
        return all_anchors
