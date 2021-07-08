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
Operators for applying image effects.
"""

from PIL import ImageOps

from mindspore.dataset.vision import py_transforms_util


class Solarize:
    """
    Solarize inverts image pixels with values above the configured threshold.

    Args:
        threshold (int): All pixels above the threshold would be inverted.
                         Ranging within [0, 255].
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to be solarized.

        Returns:
            img (PIL image), Solarized image.
        """
        if not py_transforms_util.is_pil(img):
            raise TypeError(
                py_transforms_util.augment_error_message.format(type(img)))

        return ImageOps.solarize(img, self.threshold)


class Posterize:
    """
    Posterize reduces the number of bits for each color channel.

    Args:
        bits (int): The number of bits to keep for each channel.
                    Ranging within [1, 8].
    """

    def __init__(self, bits):
        self.bits = bits

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to be posterized.

        Returns:
            img (PIL image), Posterized image.
        """
        if not py_transforms_util.is_pil(img):
            raise TypeError(
                py_transforms_util.augment_error_message.format(type(img)))

        return ImageOps.posterize(img, self.bits)
