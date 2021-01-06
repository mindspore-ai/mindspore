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

"""Anchor Generator"""

import numpy as np


class GridAnchorGenerator:
    """
    Anchor Generator
    """
    def __init__(self, image_shape, scale, scales_per_octave, aspect_ratios):
        super(GridAnchorGenerator, self).__init__()
        self.scale = scale
        self.scales_per_octave = scales_per_octave
        self.aspect_ratios = aspect_ratios
        self.image_shape = image_shape


    def generate(self, step):
        scales = np.array([2**(float(scale) / self.scales_per_octave)
                           for scale in range(self.scales_per_octave)]).astype(np.float32)
        aspects = np.array(list(self.aspect_ratios)).astype(np.float32)

        scales_grid, aspect_ratios_grid = np.meshgrid(scales, aspects)
        scales_grid = scales_grid.reshape([-1])
        aspect_ratios_grid = aspect_ratios_grid.reshape([-1])

        feature_size = [self.image_shape[0] / step, self.image_shape[1] / step]
        grid_height, grid_width = feature_size

        base_size = np.array([self.scale * step, self.scale * step]).astype(np.float32)
        anchor_offset = step / 2.0

        ratio_sqrt = np.sqrt(aspect_ratios_grid)
        heights = scales_grid / ratio_sqrt * base_size[0]
        widths = scales_grid * ratio_sqrt * base_size[1]

        y_centers = np.arange(grid_height).astype(np.float32)
        y_centers = y_centers * step + anchor_offset
        x_centers = np.arange(grid_width).astype(np.float32)
        x_centers = x_centers * step + anchor_offset
        x_centers, y_centers = np.meshgrid(x_centers, y_centers)

        x_centers_shape = x_centers.shape
        y_centers_shape = y_centers.shape

        widths_grid, x_centers_grid = np.meshgrid(widths, x_centers.reshape([-1]))
        heights_grid, y_centers_grid = np.meshgrid(heights, y_centers.reshape([-1]))

        x_centers_grid = x_centers_grid.reshape(*x_centers_shape, -1)
        y_centers_grid = y_centers_grid.reshape(*y_centers_shape, -1)
        widths_grid = widths_grid.reshape(-1, *x_centers_shape)
        heights_grid = heights_grid.reshape(-1, *y_centers_shape)


        bbox_centers = np.stack([y_centers_grid, x_centers_grid], axis=3)
        bbox_sizes = np.stack([heights_grid, widths_grid], axis=3)
        bbox_centers = bbox_centers.reshape([-1, 2])
        bbox_sizes = bbox_sizes.reshape([-1, 2])
        bbox_corners = np.concatenate([bbox_centers - 0.5 * bbox_sizes, bbox_centers + 0.5 * bbox_sizes], axis=1)
        self.bbox_corners = bbox_corners / np.array([*self.image_shape, *self.image_shape]).astype(np.float32)
        self.bbox_centers = np.concatenate([bbox_centers, bbox_sizes], axis=1)
        self.bbox_centers = self.bbox_centers / np.array([*self.image_shape, *self.image_shape]).astype(np.float32)

        print(self.bbox_centers.shape)
        return self.bbox_centers, self.bbox_corners

    def generate_multi_levels(self, steps):
        bbox_centers_list = []
        bbox_corners_list = []
        for step in steps:
            bbox_centers, bbox_corners = self.generate(step)
            bbox_centers_list.append(bbox_centers)
            bbox_corners_list.append(bbox_corners)

        self.bbox_centers = np.concatenate(bbox_centers_list, axis=0)
        self.bbox_corners = np.concatenate(bbox_corners_list, axis=0)
        return self.bbox_centers, self.bbox_corners
