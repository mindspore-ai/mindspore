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
"""FasterRcnn anchor generator."""

import numpy as np

class AnchorGenerator():
    """Anchor generator for FasterRcnn."""
    def __init__(self, base_size, scales, ratios, scale_major=True, ctr=None):
        """Anchor generator init method."""
        self.base_size = base_size
        self.scales = np.array(scales)
        self.ratios = np.array(ratios)
        self.scale_major = scale_major
        self.ctr = ctr
        self.base_anchors = self.gen_base_anchors()

    def gen_base_anchors(self):
        """Generate a single anchor."""
        w = self.base_size
        h = self.base_size
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = self.ctr

        h_ratios = np.sqrt(self.ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * self.scales[None, :]).reshape(-1)
            hs = (h * h_ratios[:, None] * self.scales[None, :]).reshape(-1)
        else:
            ws = (w * self.scales[:, None] * w_ratios[None, :]).reshape(-1)
            hs = (h * self.scales[:, None] * h_ratios[None, :]).reshape(-1)

        base_anchors = np.stack(
            [
                x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
            ],
            axis=-1).round()

        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        """Generate grid."""
        xx = np.repeat(x.reshape(1, len(x)), len(y), axis=0).reshape(-1)
        yy = np.repeat(y, len(x))
        if row_major:
            return xx, yy

        return yy, xx

    def grid_anchors(self, featmap_size, stride=16):
        """Generate anchor list."""
        base_anchors = self.base_anchors

        feat_h, feat_w = featmap_size
        shift_x = np.arange(0, feat_w) * stride
        shift_y = np.arange(0, feat_h) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = np.stack([shift_xx, shift_yy, shift_xx, shift_yy], axis=-1)
        shifts = shifts.astype(base_anchors.dtype)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.reshape(-1, 4)

        return all_anchors
