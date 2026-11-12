# Copyright 2019-2024 Huawei Technologies Co., Ltd
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
"""
Interpolation Mode, Resampling Filters
"""
import numbers
from enum import Enum, IntEnum

from PIL import Image

import mindspore._c_dataengine as cde


# The following constants have been deprecated by Pillow since version 9.1.0
if int(Image.__version__.split(".", maxsplit=1)[0]) > 9 or Image.__version__ >= "9.1.0":
    FLIP_LEFT_RIGHT = Image.Transpose.FLIP_LEFT_RIGHT
    FLIP_TOP_BOTTOM = Image.Transpose.FLIP_TOP_BOTTOM
    PERSPECTIVE = Image.Transform.PERSPECTIVE
    AFFINE = Image.Transform.AFFINE
    NEAREST = Image.Resampling.NEAREST
    ANTIALIAS = Image.Resampling.LANCZOS
    LINEAR = Image.Resampling.BILINEAR
    CUBIC = Image.Resampling.BICUBIC
else:
    FLIP_LEFT_RIGHT = Image.FLIP_LEFT_RIGHT
    FLIP_TOP_BOTTOM = Image.FLIP_TOP_BOTTOM
    PERSPECTIVE = Image.PERSPECTIVE
    AFFINE = Image.AFFINE
    NEAREST = Image.NEAREST
    ANTIALIAS = Image.ANTIALIAS
    LINEAR = Image.LINEAR
    CUBIC = Image.CUBIC




class Border(str, Enum):
    """
    Padding mode and border type.

    Possible enumeration values are: ``Border.CONSTANT``, ``Border.EDGE``, ``Border.REFLECT``, ``Border.SYMMETRIC``.

    - ``Border.CONSTANT`` : means it fills the border with constant values.
    - ``Border.EDGE`` : means it pads with the last value on the edge.
    - ``Border.REFLECT`` : means it reflects the values on the edge omitting the last value of the edge.
      For example, padding [1,2,3,4] with 2 elements on both sides will result in [3,2,1,2,3,4,3,2].
    - ``Border.SYMMETRIC`` : means it reflects the values on the edge repeating the last value of the edge.
      For example, padding [1,2,3,4] with 2 elements on both sides will result in [2,1,1,2,3,4,4,3].

    Note:
        This class is derived from class str to support JSON serialization.
    """
    CONSTANT: str = "constant"
    EDGE: str = "edge"
    REFLECT: str = "reflect"
    SYMMETRIC: str = "symmetric"

    @staticmethod
    def to_python_type(border_type):
        """
        Function to return Python type for Border Type.
        """
        python_values = {Border.CONSTANT: 'constant',
                         Border.EDGE: 'edge',
                         Border.REFLECT: 'reflect',
                         Border.SYMMETRIC: 'symmetric'}

        value = python_values.get(border_type)
        if value is None:
            raise RuntimeError("Unsupported Border type, only support CONSTANT, EDGE, REFLECT and SYMMETRIC.")
        return value

    @staticmethod
    def to_c_type(border_type):
        """
        Function to return C type for Border Type.
        """
        c_values = {Border.CONSTANT: cde.BorderType.DE_BORDER_CONSTANT,
                    Border.EDGE: cde.BorderType.DE_BORDER_EDGE,
                    Border.REFLECT: cde.BorderType.DE_BORDER_REFLECT,
                    Border.SYMMETRIC: cde.BorderType.DE_BORDER_SYMMETRIC}

        value = c_values.get(border_type)
        if value is None:
            raise RuntimeError("Unsupported Border type, only support CONSTANT, EDGE, REFLECT and SYMMETRIC.")
        return value




class ImageBatchFormat(IntEnum):
    """
    Data format of images after batch operation.

    Possible enumeration values are: ``ImageBatchFormat.NHWC``, ``ImageBatchFormat.NCHW``.

    - ``ImageBatchFormat.NHWC``: Stores data in the order of batch (N), height (H), width (W),
      and channels (C).
    - ``ImageBatchFormat.NCHW``: Stores data in the order of batch (N), channels (C), height (H),
      and width (W).
    """
    NHWC = 0
    NCHW = 1

    @staticmethod
    def to_c_type(image_batch_format):
        """
        Function to return C type for ImageBatchFormat.
        """
        c_values = {ImageBatchFormat.NHWC: cde.ImageBatchFormat.DE_IMAGE_BATCH_FORMAT_NHWC,
                    ImageBatchFormat.NCHW: cde.ImageBatchFormat.DE_IMAGE_BATCH_FORMAT_NCHW}

        value = c_values.get(image_batch_format)
        if value is None:
            raise RuntimeError("Unsupported ImageBatchFormat, only support NHWC and NCHW.")
        return value




class Inter(IntEnum):
    """
    Interpolation methods.

    Available values are as follows:

    - ``Inter.NEAREST`` : Nearest neighbor interpolation.
    - ``Inter.ANTIALIAS`` : Antialias interpolation. Supported only when the input is PIL.Image.Image.
    - ``Inter.LINEAR`` : Linear interpolation, the same as ``Inter.BILINEAR``.
    - ``Inter.BILINEAR`` : Bilinear interpolation.
    - ``Inter.CUBIC`` : Cubic interpolation, the same as ``Inter.BICUBIC``.
    - ``Inter.BICUBIC`` : Bicubic interpolation.
    - ``Inter.AREA`` : Pixel area interpolation. Supported only when the input is numpy.ndarray.
    - ``Inter.PILCUBIC`` : Pillow implementation of bicubic interpolation. Supported only when the input
      is numpy.ndarray.
    """
    NEAREST = 0
    ANTIALIAS = 1
    BILINEAR = LINEAR = 2
    BICUBIC = CUBIC = 3
    AREA = 4
    PILCUBIC = 5

    @staticmethod
    def to_python_type(inter_type):
        """
        Function to return Python type for Interpolation Mode.
        """
        python_values = {Inter.NEAREST: NEAREST,
                         Inter.ANTIALIAS: ANTIALIAS,
                         Inter.LINEAR: LINEAR,
                         Inter.CUBIC: CUBIC}

        value = python_values.get(inter_type)
        if value is None:
            raise RuntimeError("Unsupported interpolation, only support NEAREST, ANTIALIAS, LINEAR and CUBIC.")
        return value

    @staticmethod
    def to_c_type(inter_type):
        """
        Function to return C type for Interpolation Mode.
        """
        c_values = {Inter.NEAREST: cde.InterpolationMode.DE_INTER_NEAREST_NEIGHBOUR,
                    Inter.LINEAR: cde.InterpolationMode.DE_INTER_LINEAR,
                    Inter.CUBIC: cde.InterpolationMode.DE_INTER_CUBIC,
                    Inter.AREA: cde.InterpolationMode.DE_INTER_AREA,
                    Inter.PILCUBIC: cde.InterpolationMode.DE_INTER_PILCUBIC}

        value = c_values.get(inter_type)
        if value is None:
            raise RuntimeError("Unsupported interpolation, only support NEAREST, LINEAR, CUBIC, AREA and PILCUBIC.")

        return value












def parse_padding(padding):
    """ Parses and prepares the padding tuple"""

    if isinstance(padding, numbers.Number):
        padding = [padding] * 4
    if len(padding) == 2:
        left = right = padding[0]
        top = bottom = padding[1]
        padding = (left, top, right, bottom,)
    if isinstance(padding, list):
        padding = tuple(padding)
    return padding
