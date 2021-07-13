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
"""
dataset classes
"""
import cv2
import numpy as np

import src.utils.img
from src.dataset.MPIIDataLoader import flipped_parts


class GenerateHeatmap:
    """
    get train target heatmap
    """

    def __init__(self, output_res, num_parts):
        self.output_res = output_res
        self.num_parts = num_parts
        sigma = self.output_res / 64
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, keypoints):
        hms = np.zeros(shape=(self.num_parts, self.output_res, self.output_res), dtype=np.float32)
        sigma = self.sigma
        for p in keypoints:
            for idx, pt in enumerate(p):
                if pt[0] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or x >= self.output_res or y >= self.output_res:
                        continue
                    ul = int(x - 3 * sigma - 1), int(y - 3 * sigma - 1)
                    br = int(x + 3 * sigma + 2), int(y + 3 * sigma + 2)

                    c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[idx, aa:bb, cc:dd] = np.maximum(hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms


class DatasetGenerator:
    """
    mindspore general dataset generator
    """

    def __init__(self, input_res, output_res, ds, index):
        self.input_res = input_res
        self.output_res = output_res
        self.generateHeatmap = GenerateHeatmap(self.output_res, 16)
        self.ds = ds
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # print(f"loading...{idx}")
        return self.loadImage(self.index[idx])

    def loadImage(self, idx):
        """
        load and preprocess image
        """
        ds = self.ds

        # Load + Crop
        orig_img = ds.get_img(idx)
        orig_keypoints = ds.get_kps(idx)
        kptmp = orig_keypoints.copy()
        c = ds.get_center(idx)
        s = ds.get_scale(idx)

        cropped = src.utils.img.crop(orig_img, c, s, (self.input_res, self.input_res))
        for i in range(np.shape(orig_keypoints)[1]):
            if orig_keypoints[0, i, 0] > 0:
                orig_keypoints[0, i, :2] = src.utils.img.transform(
                    orig_keypoints[0, i, :2], c, s, (self.input_res, self.input_res)
                )
        keypoints = np.copy(orig_keypoints)

        # Random Crop
        height, width = cropped.shape[0:2]
        center = np.array((width / 2, height / 2))
        scale = max(height, width) / 200

        aug_rot = 0

        aug_rot = (np.random.random() * 2 - 1) * 30.0
        aug_scale = np.random.random() * (1.25 - 0.75) + 0.75
        scale *= aug_scale

        mat_mask = src.utils.img.get_transform(center, scale, (self.output_res, self.output_res), aug_rot)[:2]

        mat = src.utils.img.get_transform(center, scale, (self.input_res, self.input_res), aug_rot)[:2]
        inp = cv2.warpAffine(cropped, mat, (self.input_res, self.input_res)).astype(np.float32) / 255
        keypoints[:, :, 0:2] = src.utils.img.kpt_affine(keypoints[:, :, 0:2], mat_mask)
        if np.random.randint(2) == 0:
            inp = self.preprocess(inp)
            inp = inp[:, ::-1]
            keypoints = keypoints[:, flipped_parts["mpii"]]
            keypoints[:, :, 0] = self.output_res - keypoints[:, :, 0]
            orig_keypoints = orig_keypoints[:, flipped_parts["mpii"]]
            orig_keypoints[:, :, 0] = self.input_res - orig_keypoints[:, :, 0]

        # If keypoint is invisible, set to 0
        for i in range(np.shape(orig_keypoints)[1]):
            if kptmp[0, i, 0] == 0 and kptmp[0, i, 1] == 0:
                keypoints[0, i, 0] = 0
                keypoints[0, i, 1] = 0
                orig_keypoints[0, i, 0] = 0
                orig_keypoints[0, i, 1] = 0

        # Generate target heatmap
        heatmaps = self.generateHeatmap(keypoints)

        return inp.astype(np.float32), heatmaps.astype(np.float32)

    def preprocess(self, data):
        """
        preprocess images
        """
        # Random hue and saturation
        data = cv2.cvtColor(data, cv2.COLOR_RGB2HSV)
        delta = (np.random.random() * 2 - 1) * 0.2
        data[:, :, 0] = np.mod(data[:, :, 0] + (delta * 360 + 360.0), 360.0)

        delta_sature = np.random.random() + 0.5
        data[:, :, 1] *= delta_sature
        data[:, :, 1] = np.maximum(np.minimum(data[:, :, 1], 1), 0)
        data = cv2.cvtColor(data, cv2.COLOR_HSV2RGB)

        # Random brightness
        delta = (np.random.random() * 2 - 1) * 0.3
        data += delta

        # Random contrast
        mean = data.mean(axis=2, keepdims=True)
        data = (data - mean) * (np.random.random() + 0.5) + mean
        data = np.minimum(np.maximum(data, 0), 1)
        return data
