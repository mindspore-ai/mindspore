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
"""used in data enhance"""
import numpy as np
import cv2


class RandomStretch():
    """
    Random resize image according to the stretch
    Args:
        max_stretch(float): 0 to 1 value
    """
    def __init__(self, max_stretch=0.05):

        self.max_stretch = max_stretch #

    def __call__(self, sample):
        """
        Args:
            sample(numpy array): 3 or 1 dim image
        """
        scale_h = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        scale_w = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        h, w = sample.shape[:2]
        shape = (int(h * scale_h), int(w * scale_w))
        return cv2.resize(sample, shape, cv2.INTER_LINEAR)

class CenterCrop():
    """
    Crop the image in the center according the given size
                if size greater than image size, zero padding will adpot
            Args:
                size (tuple): desired size
    """
    def __init__(self, size):

        self.size = size   # z-> 127x127  x->255x255

    def __call__(self, sample):
        """
        Args:
            sample(numpy array): 3 or 1 dim image
        """
        shape = sample.shape[:2]
        cy, cx = (shape[0]-1) // 2, (shape[1]-1) // 2
        ymin, xmin = cy - self.size[0]//2, cx - self.size[1] // 2
        ymax, xmax = cy + self.size[0]//2 + self.size[0] % 2,\
                     cx + self.size[1]//2 + self.size[1] % 2
        left = right = top = bottom = 0
        im_h, im_w = shape
        if xmin < 0:
            left = int(abs(xmin))
        if xmax > im_w:
            right = int(xmax - im_w)
        if ymin < 0:
            top = int(abs(ymin))
        if ymax > im_h:
            bottom = int(ymax - im_h)

        xmin = int(max(0, xmin))
        xmax = int(min(im_w, xmax))
        ymin = int(max(0, ymin))
        ymax = int(min(im_h, ymax))
        im_patch = sample[ymin:ymax, xmin:xmax]
        if left != 0 or right != 0 or top != 0 or bottom != 0:
            im_patch = cv2.copyMakeBorder(im_patch, top, bottom, left, right,
                                          cv2.BORDER_CONSTANT, value=0)
        return im_patch

class RandomCrop():
    """
            Crop the image in the center according the given size
                if size greater than image size, zero padding will adpot
            Args:
                size (tuple): desired size
                max_translate: max translate of random shift
    """
    def __init__(self, size, max_translate):

        self.size = size   # 255 - 2*stride    stride=8
        self.max_translate = max_translate # 255 - 2*stride

    def __call__(self, sample):
        """
        Args:
            sample(numpy array): 3 or 1 dim image
        """
        shape = sample.shape[:2]
        cy_o = (shape[0] - 1) // 2
        cx_o = (shape[1] - 1) // 2
        cy = np.random.randint(cy_o - self.max_translate,
                               cy_o + self.max_translate+1)
        cx = np.random.randint(cx_o - self.max_translate,
                               cx_o + self.max_translate+1)
        assert abs(cy-cy_o) <= self.max_translate and \
                abs(cx-cx_o) <= self.max_translate
        ymin = cy - self.size[0] // 2
        xmin = cx - self.size[1] // 2
        ymax = cy + self.size[0] // 2 + self.size[0] % 2
        xmax = cx + self.size[1] // 2 + self.size[1] % 2
        left = right = top = bottom = 0
        im_h, im_w = shape
        if xmin < 0:
            left = int(abs(xmin))
        if xmax > im_w:
            right = int(xmax - im_w)
        if ymin < 0:
            top = int(abs(ymin))
        if ymax > im_h:
            bottom = int(ymax - im_h)

        xmin = int(max(0, xmin))
        xmax = int(min(im_w, xmax))
        ymin = int(max(0, ymin))
        ymax = int(min(im_h, ymax))
        im_patch = sample[ymin:ymax, xmin:xmax]
        if left != 0 or right != 0 or top != 0 or bottom != 0:
            im_patch = cv2.copyMakeBorder(im_patch, top, bottom, left, right,
                                          cv2.BORDER_CONSTANT, value=0)
        return im_patch

class ColorAug():
    """
        colorAug
    """
    def __init__(self, type_in='z'):
        if type_in == 'z':
            rgb_var = np.array([[3.2586416e+03, 2.8992207e+03, 2.6392236e+03],
                                [2.8992207e+03, 3.0958174e+03, 2.9321748e+03],
                                [2.6392236e+03, 2.9321748e+03, 3.4533721e+03]])
        if type_in == 'x':
            rgb_var = np.array([[2.4847285e+03, 2.1796064e+03, 1.9766885e+03],
                                [2.1796064e+03, 2.3441289e+03, 2.2357402e+03],
                                [1.9766885e+03, 2.2357402e+03, 2.7369697e+03]])
        self.v, _ = np.linalg.eig(rgb_var)
        self.v = np.sqrt(self.v)

    def __call__(self, sample):
        return sample + 0.1 * self.v * np.random.randn(3)


class RandomBlur():
    """Randomblur"""
    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, sample):
        if np.random.rand(1) < self.ratio:
            # random kernel size
            kernel_size = np.random.choice([3, 5, 7])
            # random gaussian sigma
            sigma = np.random.rand() * 5
            sample_gaussian = cv2.GaussianBlur(sample, (kernel_size, kernel_size), sigma)
        else:
            return sample
        return sample_gaussian

class Normalize():
    """
        image normalize to 0-1
    """
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, sample):
        return (sample / 255. - self.mean) / self.std

class ToTensor():
    """transpose and totensor"""
    def __call__(self, sample):
        sample = np.transpose(sample, (2, 0, 1))
        return np.array(sample, dtype=np.float32)
