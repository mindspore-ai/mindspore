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
Operators for affine transformations.
"""

import numbers
import random

from PIL import Image, __version__

from mindspore.dataset.vision.py_transforms import DE_PY_INTER_MODE
from mindspore.dataset.vision.py_transforms_util import (
    augment_error_message,
    is_pil,
    rotate,
)
from mindspore.dataset.vision.utils import Inter


class ShearX:
    """
    ShearX shears images along the x-axis.

    Args:
        shear (int): the pixel size to shear.
        resample (enum): the interpolation mode.
        fill_value (int or tuple): the filling value to fill the area outside
                                   the transform in the output image.
    """

    def __init__(self, shear, resample=Inter.NEAREST, fill_value=0):
        if not isinstance(shear, numbers.Number):
            raise TypeError('shear must be a single number.')

        self.shear = shear
        self.resample = DE_PY_INTER_MODE[resample]
        self.fill_value = fill_value

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to apply shear_x transformation.

        Returns:
            img (PIL image), X-axis sheared image.
        """
        if not is_pil(img):
            raise ValueError('Input image should be a Pillow image.')

        output_size = img.size
        shear = self.shear if random.random() > 0.5 else -self.shear
        matrix = (1, shear, 0, 0, 1, 0)

        if __version__ >= '5':
            kwargs = {'fillcolor': self.fill_value}
        else:
            kwargs = {}

        return img.transform(output_size, Image.AFFINE, matrix,
                             self.resample, **kwargs)


class ShearY:
    """
    ShearY shears images along the y-axis.

    Args:
        shear (int): the pixel size to shear.
        resample (enum): the interpolation mode.
        fill_value (int or tuple): the filling value to fill the area outside
                                   the transform in the output image.
    """

    def __init__(self, shear, resample=Inter.NEAREST, fill_value=0):
        if not isinstance(shear, numbers.Number):
            raise TypeError('shear must be a single number.')

        self.shear = shear
        self.resample = DE_PY_INTER_MODE[resample]
        self.fill_value = fill_value

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to apply shear_y transformation.

        Returns:
            img (PIL image), Y-axis sheared image.
        """
        if not is_pil(img):
            raise ValueError('Input image should be a Pillow image.')

        output_size = img.size
        shear = self.shear if random.random() > 0.5 else -self.shear
        matrix = (1, 0, 0, shear, 1, 0)

        if __version__ >= '5':
            kwargs = {'fillcolor': self.fill_value}
        else:
            kwargs = {}

        return img.transform(output_size, Image.AFFINE, matrix,
                             self.resample, **kwargs)


class TranslateX:
    """
    TranslateX translates images along the x-axis.

    Args:
        translate (int): the pixel size to translate.
        resample (enum): the interpolation mode.
        fill_value (int or tuple): the filling value to fill the area outside
                                   the transform in the output image.
    """

    def __init__(self, translate, resample=Inter.NEAREST, fill_value=0):
        if not isinstance(translate, numbers.Number):
            raise TypeError('translate must be a single number.')

        self.translate = translate
        self.resample = DE_PY_INTER_MODE[resample]
        self.fill_value = fill_value

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to apply translate_x transformation.

        Returns:
            img (PIL image), X-axis translated image.
        """
        if not is_pil(img):
            raise ValueError('Input image should be a Pillow image.')

        output_size = img.size
        trans = self.translate if random.random() > 0.5 else -self.translate
        matrix = (1, 0, trans, 0, 1, 0)

        if __version__ >= '5':
            kwargs = {'fillcolor': self.fill_value}
        else:
            kwargs = {}

        return img.transform(output_size, Image.AFFINE, matrix,
                             self.resample, **kwargs)


class TranslateY:
    """
    TranslateY translates images along the y-axis.

    Args:
        translate (int): the pixel size to translate.
        resample (enum): the interpolation mode.
        fill_value (int or tuple): the filling value to fill the area outside
                                   the transform in the output image.
    """

    def __init__(self, translate, resample=Inter.NEAREST, fill_value=0):
        if not isinstance(translate, numbers.Number):
            raise TypeError('Translate must be a single number.')

        self.translate = translate
        self.resample = DE_PY_INTER_MODE[resample]
        self.fill_value = fill_value

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to apply translate_y transformation.

        Returns:
            img (PIL image), Y-axis translated image.
        """
        if not is_pil(img):
            raise ValueError('Input image should be a Pillow image.')

        output_size = img.size
        trans = self.translate if random.random() > 0.5 else -self.translate
        matrix = (1, 0, 0, 0, 1, trans)

        if __version__ >= '5':
            kwargs = {'fillcolor': self.fill_value}
        else:
            kwargs = {}

        return img.transform(output_size, Image.AFFINE, matrix,
                             self.resample, **kwargs)


class Rotate:
    """
    Rotate is similar to py_vision.RandomRotation but uses a fixed degree.

    Args:
        degree (int): the degree to rotate.

    Please refer to py_transforms.RandomRotation for more argument
    specifications.
    """

    def __init__(
            self, degree,
            resample=Inter.NEAREST, expand=False, center=None, fill_value=0,
    ):
        if not isinstance(degree, numbers.Number):
            raise TypeError('degree must be a single number.')

        self.degree = degree
        self.resample = DE_PY_INTER_MODE[resample]
        self.expand = expand
        self.center = center
        self.fill_value = fill_value

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to be rotated.

        Returns:
            img (PIL image), Rotated image.
        """
        if not is_pil(img):
            raise TypeError(augment_error_message.format(type(img)))

        degree = self.degree if random.random() > 0.5 else -self.degree
        return rotate(img, degree, self.resample, self.expand,
                      self.center, self.fill_value)
