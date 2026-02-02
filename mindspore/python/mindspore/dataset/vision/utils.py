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
import gc
import importlib
import math
import numbers
import os
import re
from enum import Enum, IntEnum
from fractions import Fraction

import numpy as np
from PIL import Image

import mindspore
import mindspore._c_dataengine as cde
from mindspore import log as logger
from mindspore.dataset.core.validator_helpers import check_file, check_value, type_check, type_check_list
from ..core.config import get_video_backend

_CALLED_TIMES = 0
_GC_COLLECTION_INTERVAL = 10
_INITIALIZED = False
_INITIALIZED_PID = False

# The following constants have been deprecated by Pillow since version 9.1.0
if int(Image.__version__.split(".")[0]) > 9 or Image.__version__ >= "9.1.0":
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


class AutoAugmentPolicy(str, Enum):
    """
    AutoAugment policy for different datasets.

    Possible enumeration values are: ``AutoAugmentPolicy.IMAGENET``, ``AutoAugmentPolicy.CIFAR10``,
    ``AutoAugmentPolicy.SVHN``.

    Each policy contains 25 pairs of augmentation operations. When using AutoAugment, each image is randomly
    transformed with one of these operation pairs. Each pair has 2 different operations. The following shows
    all of these augmentation operations, including operation names with their probabilities and random
    parameters.

    - ``AutoAugmentPolicy.IMAGENET``: dataset auto augment policy for ImageNet.

      .. code-block::

          Augmentation operations pair:
          [(("Posterize", 0.4, 8), ("Rotate", 0.6, 9)),        (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
           (("Equalize", 0.8, None), ("Equalize", 0.6, None)), (("Posterize", 0.6, 7), ("Posterize", 0.6, 6)),
           (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),    (("Equalize", 0.4, None), ("Rotate", 0.8, 8)),
           (("Solarize", 0.6, 3), ("Equalize", 0.6, None)),    (("Posterize", 0.8, 5), ("Equalize", 1.0, None)),
           (("Rotate", 0.2, 3), ("Solarize", 0.6, 8)),         (("Equalize", 0.6, None), ("Posterize", 0.4, 6)),
           (("Rotate", 0.8, 8), ("Color", 0.4, 0)),            (("Rotate", 0.4, 9), ("Equalize", 0.6, None)),
           (("Equalize", 0.0, None), ("Equalize", 0.8, None)), (("Invert", 0.6, None), ("Equalize", 1.0, None)),
           (("Color", 0.6, 4), ("Contrast", 1.0, 8)),          (("Rotate", 0.8, 8), ("Color", 1.0, 2)),
           (("Color", 0.8, 8), ("Solarize", 0.8, 7)),          (("Sharpness", 0.4, 7), ("Invert", 0.6, None)),
           (("ShearX", 0.6, 5), ("Equalize", 1.0, None)),      (("Color", 0.4, 0), ("Equalize", 0.6, None)),
           (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),    (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
           (("Invert", 0.6, None), ("Equalize", 1.0, None)),   (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
           (("Equalize", 0.8, None), ("Equalize", 0.6, None))]

    - ``AutoAugmentPolicy.CIFAR10``: dataset auto augment policy for Cifar10.

      .. code-block::

          Augmentation operations pair:
          [(("Invert", 0.1, None), ("Contrast", 0.2, 6)),         (("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)),
           (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)),         (("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)),
           (("AutoContrast", 0.5, None), ("Equalize", 0.9, None)), (("ShearY", 0.2, 7), ("Posterize", 0.3, 7)),
           (("Color", 0.4, 3), ("Brightness", 0.6, 7)),            (("Sharpness", 0.3, 9), ("Brightness", 0.7, 9)),
           (("Equalize", 0.6, None), ("Equalize", 0.5, None)),     (("Contrast", 0.6, 7), ("Sharpness", 0.6, 5)),
           (("Color", 0.7, 7), ("TranslateX", 0.5, 8)),            (("Equalize", 0.8, None), ("Invert", 0.1, None)),
           (("TranslateY", 0.4, 3), ("Sharpness", 0.2, 6)),        (("Brightness", 0.9, 6), ("Color", 0.2, 8)),
           (("Solarize", 0.5, 2), ("Invert", 0.0, None)),          (("TranslateY", 0.9, 9), ("TranslateY", 0.7, 9)),
           (("Equalize", 0.2, None), ("Equalize", 0.6, None)),     (("Color", 0.9, 9), ("Equalize", 0.6, None)),
           (("AutoContrast", 0.8, None), ("Solarize", 0.2, 8)),    (("Brightness", 0.1, 3), ("Color", 0.7, 0)),
           (("Solarize", 0.4, 5), ("AutoContrast", 0.9, None)),
           (("AutoContrast", 0.9, None), ("Solarize", 0.8, 3)),
           (("TranslateY", 0.7, 9), ("AutoContrast", 0.9, None)),
           (("Equalize", 0.3, None), ("AutoContrast", 0.4, None)),
           (("Equalize", 0.2, None), ("AutoContrast", 0.6, None))]

    - ``AutoAugmentPolicy.SVHN``: dataset auto augment policy for SVHN.

      .. code-block::

          Augmentation operations pair:
          [(("ShearX", 0.9, 4), ("Invert", 0.2, None)),          (("ShearY", 0.9, 8), ("Invert", 0.7, None)),
           (("Equalize", 0.6, None), ("Solarize", 0.6, 6)),      (("Invert", 0.9, None), ("Equalize", 0.6, None)),
           (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),        (("ShearX", 0.9, 4), ("AutoContrast", 0.8, None)),
           (("ShearY", 0.9, 8), ("Invert", 0.4, None)),          (("ShearY", 0.9, 5), ("Solarize", 0.2, 6)),
           (("Invert", 0.9, None), ("AutoContrast", 0.8, None)), (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
           (("ShearX", 0.9, 4), ("Solarize", 0.3, 3)),           (("ShearY", 0.8, 8), ("Invert", 0.7, None)),
           (("Equalize", 0.9, None), ("TranslateY", 0.6, 6)),    (("Invert", 0.9, None), ("Equalize", 0.6, None)),
           (("Contrast", 0.3, 3), ("Rotate", 0.8, 4)),           (("Invert", 0.8, None), ("TranslateY", 0.0, 2)),
           (("ShearY", 0.7, 6), ("Solarize", 0.4, 8)),           (("Invert", 0.6, None), ("Rotate", 0.8, 4)),
           (("ShearY", 0.3, 7), ("TranslateX", 0.9, 3)),         (("ShearX", 0.1, 6), ("Invert", 0.6, None)),
           (("Solarize", 0.7, 2), ("TranslateY", 0.6, 7)),       (("ShearY", 0.8, 4), ("Invert", 0.8, None)),
           (("ShearX", 0.7, 9), ("TranslateY", 0.8, 3)),         (("ShearY", 0.8, 5), ("AutoContrast", 0.7, None)),
           (("ShearX", 0.7, 2), ("Invert", 0.1, None))]
    """
    IMAGENET: str = "imagenet"
    CIFAR10: str = "cifar10"
    SVHN: str = "svhn"

    @staticmethod
    def to_c_type(policy):
        """
        Function to return C type for AutoAugment policy.
        """
        c_values = {AutoAugmentPolicy.IMAGENET: cde.AutoAugmentPolicy.DE_AUTO_AUGMENT_POLICY_IMAGENET,
                    AutoAugmentPolicy.CIFAR10: cde.AutoAugmentPolicy.DE_AUTO_AUGMENT_POLICY_CIFAR10,
                    AutoAugmentPolicy.SVHN: cde.AutoAugmentPolicy.DE_AUTO_AUGMENT_POLICY_SVHN}

        value = c_values.get(policy)
        if value is None:
            raise RuntimeError("Unsupported AutoAugmentPolicy, only support IMAGENET, CIFAR10, and SVHN.")
        return value


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


class ConvertMode(IntEnum):
    """
    The color conversion mode.

    Possible enumeration values are as follows:

    - ConvertMode.COLOR_BGR2BGRA: convert BGR format images to BGRA format images.
    - ConvertMode.COLOR_RGB2RGBA: convert RGB format images to RGBA format images.
    - ConvertMode.COLOR_BGRA2BGR: convert BGRA format images to BGR format images.
    - ConvertMode.COLOR_RGBA2RGB: convert RGBA format images to RGB format images.
    - ConvertMode.COLOR_BGR2RGBA: convert BGR format images to RGBA format images.
    - ConvertMode.COLOR_RGB2BGRA: convert RGB format images to BGRA format images.
    - ConvertMode.COLOR_RGBA2BGR: convert RGBA format images to BGR format images.
    - ConvertMode.COLOR_BGRA2RGB: convert BGRA format images to RGB format images.
    - ConvertMode.COLOR_BGR2RGB: convert BGR format images to RGB format images.
    - ConvertMode.COLOR_RGB2BGR: convert RGB format images to BGR format images.
    - ConvertMode.COLOR_BGRA2RGBA: convert BGRA format images to RGBA format images.
    - ConvertMode.COLOR_RGBA2BGRA: convert RGBA format images to BGRA format images.
    - ConvertMode.COLOR_BGR2GRAY: convert BGR format images to GRAY format images.
    - ConvertMode.COLOR_RGB2GRAY: convert RGB format images to GRAY format images.
    - ConvertMode.COLOR_GRAY2BGR: convert GRAY format images to BGR format images.
    - ConvertMode.COLOR_GRAY2RGB: convert GRAY format images to RGB format images.
    - ConvertMode.COLOR_GRAY2BGRA: convert GRAY format images to BGRA format images.
    - ConvertMode.COLOR_GRAY2RGBA: convert GRAY format images to RGBA format images.
    - ConvertMode.COLOR_BGRA2GRAY: convert BGRA format images to GRAY format images.
    - ConvertMode.COLOR_RGBA2GRAY: convert RGBA format images to GRAY format images.
    """
    COLOR_BGR2BGRA = 0
    COLOR_RGB2RGBA = COLOR_BGR2BGRA
    COLOR_BGRA2BGR = 1
    COLOR_RGBA2RGB = COLOR_BGRA2BGR
    COLOR_BGR2RGBA = 2
    COLOR_RGB2BGRA = COLOR_BGR2RGBA
    COLOR_RGBA2BGR = 3
    COLOR_BGRA2RGB = COLOR_RGBA2BGR
    COLOR_BGR2RGB = 4
    COLOR_RGB2BGR = COLOR_BGR2RGB
    COLOR_BGRA2RGBA = 5
    COLOR_RGBA2BGRA = COLOR_BGRA2RGBA
    COLOR_BGR2GRAY = 6
    COLOR_RGB2GRAY = 7
    COLOR_GRAY2BGR = 8
    COLOR_GRAY2RGB = COLOR_GRAY2BGR
    COLOR_GRAY2BGRA = 9
    COLOR_GRAY2RGBA = COLOR_GRAY2BGRA
    COLOR_BGRA2GRAY = 10
    COLOR_RGBA2GRAY = 11

    @staticmethod
    def to_c_type(mode):
        """
        Function to return C type for color mode.
        """
        c_values = {ConvertMode.COLOR_BGR2BGRA: cde.ConvertMode.DE_COLOR_BGR2BGRA,
                    ConvertMode.COLOR_RGB2RGBA: cde.ConvertMode.DE_COLOR_RGB2RGBA,
                    ConvertMode.COLOR_BGRA2BGR: cde.ConvertMode.DE_COLOR_BGRA2BGR,
                    ConvertMode.COLOR_RGBA2RGB: cde.ConvertMode.DE_COLOR_RGBA2RGB,
                    ConvertMode.COLOR_BGR2RGBA: cde.ConvertMode.DE_COLOR_BGR2RGBA,
                    ConvertMode.COLOR_RGB2BGRA: cde.ConvertMode.DE_COLOR_RGB2BGRA,
                    ConvertMode.COLOR_RGBA2BGR: cde.ConvertMode.DE_COLOR_RGBA2BGR,
                    ConvertMode.COLOR_BGRA2RGB: cde.ConvertMode.DE_COLOR_BGRA2RGB,
                    ConvertMode.COLOR_BGR2RGB: cde.ConvertMode.DE_COLOR_BGR2RGB,
                    ConvertMode.COLOR_RGB2BGR: cde.ConvertMode.DE_COLOR_RGB2BGR,
                    ConvertMode.COLOR_BGRA2RGBA: cde.ConvertMode.DE_COLOR_BGRA2RGBA,
                    ConvertMode.COLOR_RGBA2BGRA: cde.ConvertMode.DE_COLOR_RGBA2BGRA,
                    ConvertMode.COLOR_BGR2GRAY: cde.ConvertMode.DE_COLOR_BGR2GRAY,
                    ConvertMode.COLOR_RGB2GRAY: cde.ConvertMode.DE_COLOR_RGB2GRAY,
                    ConvertMode.COLOR_GRAY2BGR: cde.ConvertMode.DE_COLOR_GRAY2BGR,
                    ConvertMode.COLOR_GRAY2RGB: cde.ConvertMode.DE_COLOR_GRAY2RGB,
                    ConvertMode.COLOR_GRAY2BGRA: cde.ConvertMode.DE_COLOR_GRAY2BGRA,
                    ConvertMode.COLOR_GRAY2RGBA: cde.ConvertMode.DE_COLOR_GRAY2RGBA,
                    ConvertMode.COLOR_BGRA2GRAY: cde.ConvertMode.DE_COLOR_BGRA2GRAY,
                    ConvertMode.COLOR_RGBA2GRAY: cde.ConvertMode.DE_COLOR_RGBA2GRAY,
                    }

        mode = c_values.get(mode)
        if mode is None:
            raise RuntimeError("Unsupported ConvertMode, see https://www.mindspore.cn/docs/en/master/api_python/"
                               "dataset_vision/mindspore.dataset.vision.ConvertColor.html for more details.")
        return mode


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


class ImageReadMode(IntEnum):
    """
    The read mode used for the image file.

    Possible enumeration values are: ``ImageReadMode.UNCHANGED``, ``ImageReadMode.GRAYSCALE``, ``ImageReadMode.COLOR``.

    - ``ImageReadMode.UNCHANGED``: remain the output in the original format.
    - ``ImageReadMode.GRAYSCALE``: convert the output into one channel grayscale data.
    - ``ImageReadMode.COLOR``: convert the output into three channels RGB color data.
    """
    UNCHANGED = 0
    GRAYSCALE = 1
    COLOR = 2

    @staticmethod
    def to_c_type(image_read_mode):
        """
        Function to return C type for ImageReadMode.
        """
        c_values = {ImageReadMode.UNCHANGED: cde.ImageReadMode.DE_IMAGE_READ_MODE_UNCHANGED,
                    ImageReadMode.GRAYSCALE: cde.ImageReadMode.DE_IMAGE_READ_MODE_GRAYSCALE,
                    ImageReadMode.COLOR: cde.ImageReadMode.DE_IMAGE_READ_MODE_COLOR}

        value = c_values.get(image_read_mode)
        if value is None:
            raise RuntimeError("Unsupported ImageReadMode, only support UNCHANGED, GRAYSCALE and COLOR.")
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


class SliceMode(IntEnum):
    """
    Mode for slicing a tensor into multiple parts.

    Possible enumeration values are: ``SliceMode.PAD``, ``SliceMode.DROP``.

    - ``SliceMode.PAD``: Pads some pixels before slicing the tensor if needed.
    - ``SliceMode.DROP``: Drops remaining pixels before slicing the tensor if needed.
    """
    PAD = 0
    DROP = 1

    @staticmethod
    def to_c_type(mode):
        """
        Function to return C type for SliceMode.
        """
        c_values = {SliceMode.PAD: cde.SliceMode.DE_SLICE_PAD,
                    SliceMode.DROP: cde.SliceMode.DE_SLICE_DROP}

        value = c_values.get(mode)
        if value is None:
            raise RuntimeError("Unsupported SliceMode, only support PAD and DROP.")
        return value


def encode_jpeg(image, quality=75):
    """
    Encode the input image as JPEG data.

    Args:
        image (Union[numpy.ndarray, mindspore.Tensor]): The image to be encoded.
        quality (int, optional): Quality of the resulting JPEG data, in the range [1, 100].
            Default: ``75``.

    Returns:
        numpy.ndarray, one-dimensional uint8 data.

    Raises:
        TypeError: If `image` is not of type numpy.ndarray or mindspore.Tensor.
        TypeError: If `quality` is not of type int.
        RuntimeError: If the data type of `image` is not uint8.
        RuntimeError: If the shape of `image` is not <H, W> or <H, W, 1> or <H, W, 3>.
        RuntimeError: If `quality` is less than 1 or greater than 100.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import mindspore.dataset.vision as vision
        >>> import numpy as np
        >>> # Generate a random image with height=120, width=340, channels=3
        >>> image = np.random.randint(256, size=(120, 340, 3), dtype=np.uint8)
        >>> jpeg_data = vision.encode_jpeg(image)
    """
    if not isinstance(quality, int):
        raise TypeError("Input quality is not of type {0}, but got: {1}.".format(int, type(quality)))
    if isinstance(image, np.ndarray):
        return cde.encode_jpeg(cde.Tensor(image), quality).as_array()
    if isinstance(image, mindspore.Tensor):
        return cde.encode_jpeg(cde.Tensor(image.asnumpy()), quality).as_array()
    raise TypeError("Input image is not of type {0} or {1}, but got: {2}.".format(np.ndarray,
                                                                                  mindspore.Tensor, type(image)))


def encode_png(image, compression_level=6):
    """
    Encode the input image as PNG data.

    Args:
        image (Union[numpy.ndarray, mindspore.Tensor]): The image to be encoded.
        compression_level (int, optional): Compression level for encoding, in the range [0, 9].
            Default: ``6``.

    Returns:
        numpy.ndarray, one-dimensional uint8 data.

    Raises:
        TypeError: If `image` is not of type numpy.ndarray or mindspore.Tensor.
        TypeError: If `compression_level` is not of type int.
        RuntimeError: If the data type of `image` is not uint8.
        RuntimeError: If the shape of `image` is not <H, W> or <H, W, 1> or <H, W, 3>.
        RuntimeError: If `compression_level` is less than 0 or greater than 9.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import mindspore.dataset.vision as vision
        >>> import numpy as np
        >>> # Generate a random image with height=120, width=340, channels=3
        >>> image = np.random.randint(256, size=(120, 340, 3), dtype=np.uint8)
        >>> png_data = vision.encode_png(image)
    """
    if not isinstance(compression_level, int):
        raise TypeError("Input compression_level is not of type {0}, but got: {1}.".format(int,
                                                                                           type(compression_level)))
    if isinstance(image, np.ndarray):
        return cde.encode_png(cde.Tensor(image), compression_level).as_array()
    if isinstance(image, mindspore.Tensor):
        return cde.encode_png(cde.Tensor(image.asnumpy()), compression_level).as_array()
    raise TypeError("Input image is not of type {0} or {1}, but got: {2}.".format(np.ndarray,
                                                                                  mindspore.Tensor, type(image)))


def get_image_num_channels(image):
    """
    Get the number of input image channels.

    Args:
        image (Union[numpy.ndarray, PIL.Image.Image]): The image from which to get the number of
            channels.

    Returns:
        int, the number of input image channels.

    Raises:
        RuntimeError: If the number of dimensions of `image` is less than 2.
        TypeError: If `image` is not of type numpy.ndarray or PIL Image.

    Examples:
        >>> import mindspore.dataset.vision as vision
        >>> from PIL import Image
        >>> image = Image.open("/path/to/image_file")
        >>> num_channels = vision.get_image_num_channels(image)
    """

    if isinstance(image, np.ndarray):
        return cde.get_image_num_channels(cde.Tensor(image))

    if isinstance(image, Image.Image):
        if hasattr(image, "getbands"):
            return len(image.getbands())

        return image.channels

    raise TypeError("Input image is not of type {0} or {1}, but got: {2}.".format(np.ndarray, Image.Image, type(image)))


def get_image_size(image):
    """
    Get the size of input image as [height, width].

    Args:
        image (Union[numpy.ndarray, PIL.Image.Image]): The image from which to get the size.

    Returns:
        list[int, int], the image size.

    Raises:
        RuntimeError: If the number of dimensions of `image` is less than 2.
        TypeError: If `image` is not of type numpy.ndarray or PIL Image.

    Examples:
        >>> import mindspore.dataset.vision as vision
        >>> from PIL import Image
        >>> image = Image.open("/path/to/image_file")
        >>> image_size = vision.get_image_size(image)
    """

    if isinstance(image, np.ndarray):
        return cde.get_image_size(cde.Tensor(image))
    if isinstance(image, Image.Image):
        size_list = list(image.size)
        size_list[0], size_list[1] = size_list[1], size_list[0]
        return size_list

    raise TypeError("Input image is not of type {0} or {1}, but got: {2}.".format(np.ndarray, Image.Image, type(image)))


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


def read_file(filename):
    """
    Read a file in binary mode.

    Args:
        filename (str): The path to the file to be read.

    Returns:
        numpy.ndarray, one-dimensional uint8 data.

    Raises:
        TypeError: If `filename` is not of type str.
        RuntimeError: If `filename` does not exist or is not a regular file.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import mindspore.dataset.vision as vision
        >>> output = vision.read_file("/path/to/file")
    """
    if isinstance(filename, str):
        return cde.read_file(filename).as_array()
    raise TypeError("Input filename is not of type {0}, but got: {1}.".format(str, type(filename)))


def read_image(filename, mode=ImageReadMode.UNCHANGED):
    """
    Read an image file and decode it into single-channel grayscale data or RGB color data.
    Supported file types are JPEG, PNG, BMP, TIFF.

    Args:
        filename (str): The path to the image file to be read.
        mode (ImageReadMode, optional): The mode used for decoding the image. It can be
            ``ImageReadMode.UNCHANGED``, ``ImageReadMode.GRAYSCALE``, ``ImageReadMode.COLOR``.
            Default: ``ImageReadMode.UNCHANGED``.

            - ImageReadMode.UNCHANGED, remain the output in the original format.

            - ImageReadMode.GRAYSCALE, convert the output into one channel grayscale data.

            - ImageReadMode.COLOR, convert the output into three channels RGB color data.

    Returns:
        numpy.ndarray, three-dimensional uint8 data in the shape of (Height, Width, Channels).

    Raises:
        TypeError: If `filename` is not of type str.
        TypeError: If `mode` is not of type :class:`mindspore.dataset.vision.ImageReadMode` .
        RuntimeError: If `filename` does not exist, or is not a regular file, or is not a
            supported image file.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import mindspore.dataset.vision as vision
        >>> from mindspore.dataset.vision import ImageReadMode
        >>> output = vision.read_image("/path/to/image_file", ImageReadMode.UNCHANGED)
    """
    if not isinstance(filename, str):
        raise TypeError("Input filename is not of type {0}, but got: {1}.".format(str, type(filename)))
    if not isinstance(mode, ImageReadMode):
        raise TypeError("Input mode is not of type {0}, but got: {1}.".format(ImageReadMode, type(mode)))
    return cde.read_image(filename, ImageReadMode.to_c_type(mode)).as_array()


class DecodeParams:
    """ Struct to store decoder parameters. """

    def __init__(self, container, start_offset, end_offset, stream):
        self.container = container
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.stream = stream


class VideoFrameDvpp:
    """ Struct to store parameters of decoder. """

    dts: int
    pts: int
    positions: int
    frame: np.ndarray

    def __init__(self, dts=0, pts=0):
        self.pts = pts
        self.dts = dts


def _get_frame_by_cv(filename, container, stream):
    """ Grab video frames with OpenCV. """

    try:
        cv2 = importlib.import_module("cv2")
    except ModuleNotFoundError:
        raise ImportError("Importing cv2 failed, try to install it by running `pip install opencv-python`.")

    cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_FORMAT, -1)
    frames = {}
    pts_list = []
    pts_per_frame = round(1 / cap.get(cv2.CAP_PROP_POS_AVI_RATIO) / cap.get(cv2.CAP_PROP_FPS), 0)

    for packet in container.demux(stream):
        if packet.pts is not None:
            frame = VideoFrameDvpp(packet.dts, packet.pts)
            _, frame.frame = cap.read()
            pts_list.append(packet.pts)
            frames[frame.pts] = frame
    cap.release()
    pts_list.sort()
    position_list = {value: index for index, value in enumerate(pts_list)}
    for frame in frames.values():
        frame.positions = position_list[frame.pts]

    return frames, pts_per_frame


def _align_audio_frames(aframes, audio_frames, ref_start, ref_end):
    """ Align audio frames with specified start and end. """

    start, end = audio_frames[0].pts, audio_frames[-1].pts
    total_aframes = aframes.shape[1]
    step_per_aframe = (end - start + 1) / total_aframes
    s_idx = 0
    e_idx = total_aframes
    if start < ref_start:
        s_idx = int((ref_start - start) / step_per_aframe)
    if end > ref_end:
        e_idx = int((ref_end - end) / step_per_aframe)
    return aframes[:, s_idx:e_idx]


def _decode_video_dvpp(decode_params, frames, pts_per_frame):
    """ Send frames to Ascend and using DVPP to decode. """

    container = decode_params.container
    start_offset = decode_params.start_offset
    end_offset = decode_params.end_offset
    stream = decode_params.stream

    codecs_type = stream.name
    hi_pt_h264 = 96
    hi_pt_h265 = 265
    if codecs_type == "h264":
        codec_id = hi_pt_h264
    elif codecs_type == "hevc":
        codec_id = hi_pt_h265
    else:
        raise ValueError(f"The video codecs_type should be either 'h264' or 'hevc', got {codecs_type}.")

    # if start_offset is between 2 frames, get one more previous frame
    start_offset = int(start_offset / pts_per_frame) * pts_per_frame

    frame_width = stream.width
    frame_height = stream.height

    # update end_offset_real
    end_offset_real = end_offset
    # if end_offset equals to a frame's pts, get one more this frame
    if end_offset_real % pts_per_frame == 0:
        end_offset_real += 1
    end_offset_real = min(end_offset_real, len(frames) * pts_per_frame)
    start_frame = math.ceil(start_offset / pts_per_frame)
    total_frame = math.ceil((end_offset_real - start_offset) / pts_per_frame)

    if end_offset_real < start_offset or total_frame == 0:
        return np.empty(0, dtype=np.uint8)
    ret_tensor = cde.DeviceBuffer([total_frame, 3, frame_height, frame_width])

    # decode from dvpp
    chn = cde.decode_video_create_chn(codec_id)
    cde.decode_video_start_get_frame(chn, total_frame)

    for packet in container.demux(stream):
        if packet.pts is not None:
            frame = frames[packet.pts].frame
            input_tensor = cde.DeviceBuffer.from_numpy(frame)

            if start_offset <= int(packet.pts) <= end_offset:
                display = True
                output_tensor = ret_tensor[frames[packet.pts].positions - start_frame]
            else:
                display = False
                output_tensor = cde.DeviceBuffer([])
            # 12:rgb888packed; 13:bgr888packed; 69:rgb888planer; 70:bgr888planer. Packed is HWC, planer is CHW
            # use CHW to avoid memory copy
            cde.decode_video_send_stream(chn, input_tensor, 69, display, output_tensor)

    # ret_tensor is ordered by pts
    ret_tensor_dvpp = cde.decode_video_stop_get_frame(chn, total_frame)

    # if ret_tensor_dvpp empty, means ret_tensor already filled
    if ret_tensor_dvpp.size() != 0:
        ret_tensor = ret_tensor_dvpp

    ret_numpy = ret_tensor.numpy()

    cde.decode_video_destroy_chnl(chn)
    return ret_numpy


def _check_buffer(extradata):
    """ Check if the video should be buffered. """

    should_buffer = True
    if extradata and b"DivX" in extradata:
        # can't use regex directly because of some weird characters sometimes...
        pos = extradata.find(b"DivX")
        d = extradata[pos:]
        o = re.search(rb"DivX(\d+)Build(\d+)(\w)", d)
        if o is None:
            o = re.search(rb"DivX(\d+)b(\d+)(\w)", d)
        if o is not None:
            should_buffer = o.group(3) == b"p"
    return should_buffer


def _read_from_stream_dvpp(filename, container, start_offset, end_offset, pts_unit, stream, stream_name):
    """ Read video stream with DVPP. """

    if not stream.type == "video":
        raise RuntimeError("_read_from_stream_dvpp only handle video type")
    if pts_unit == "sec" and stream.time_base != 0:
        start_offset = int(math.floor(start_offset * (1 / stream.time_base)))
        if end_offset != float("inf") and stream.time_base != 0:
            end_offset = int(math.ceil(end_offset * (1 / stream.time_base)))
    else:
        logger.warning("The pts_unit 'pts' gives wrong results. Please use pts_unit 'sec'.")

    max_buffer_size = 5
    # DivX-style packed B-frames can have out-of-order pts (2 frames in a single pkt)
    # so need to buffer some extra frames to sort everything properly
    should_buffer = _check_buffer(stream.codec_context.extradata)

    seek_offset = start_offset
    # some files don't seek to the right location, so better be safe here
    seek_offset = max(seek_offset - 1, 0)

    if should_buffer:
        seek_offset = max(seek_offset - max_buffer_size, 0)
    # init frames before seek
    frames, pts_per_frame = _get_frame_by_cv(filename, container, stream)
    container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    decode_params = DecodeParams(container, start_offset, end_offset, stream)
    frames = _decode_video_dvpp(decode_params, frames, pts_per_frame)

    if frames is None:
        logger.warning(f"_decode_video_dvpp failed: {filename}")

    return frames


def _read_from_stream_ffmpeg(container, start_offset, end_offset, pts_unit, stream, stream_name):
    """ Read video stream with FFMPEG. """

    global _CALLED_TIMES, _GC_COLLECTION_INTERVAL
    _CALLED_TIMES += 1
    if _CALLED_TIMES % _GC_COLLECTION_INTERVAL == _GC_COLLECTION_INTERVAL - 1:
        gc.collect()

    if pts_unit == "sec":
        # sec and convert to MS in C++
        start_offset = int(math.floor(start_offset * (1 / stream.time_base)))
        if end_offset != float("inf"):
            end_offset = int(math.ceil(end_offset * (1 / stream.time_base)))
    else:
        logger.warning("The pts_unit 'pts' gives wrong results. Please use pts_unit 'sec'.")

    frames = {}
    max_buffer_size = 5
    should_buffer = True
    if stream.type == "video":
        # DivX-style packed B-frames can have out-of-order pts (2 frames in a single pkt)
        # so need to buffer some extra frames to sort everything properly
        should_buffer = _check_buffer(stream.codec_context.extradata)

    seek_offset = start_offset
    # some files don't seek to the right location, so better be safe here
    seek_offset = max(seek_offset - 1, 0)
    if should_buffer:
        seek_offset = max(seek_offset - max_buffer_size, 0)
    container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    buffer_count = 0

    for frame in container.decode(**stream_name):
        frames[frame.pts] = frame
        if frame.pts >= end_offset:
            if should_buffer and buffer_count < max_buffer_size:
                buffer_count += 1
                continue
            break

    # ensure that the results are sorted with the pts
    result = [frames[i] for i in sorted(frames) if start_offset <= frames[i].pts <= end_offset]
    if frames and start_offset > 0 and start_offset not in frames:
        # if there is no frame that exactly matches the pts of start_offset
        # add the last frame smaller than start_offset, to guarantee that
        # we will have all the necessary data. This is most useful for audio
        preceding_frames = [i for i in frames if i < start_offset]
        if preceding_frames:
            first_frame_pts = max(preceding_frames)
            result.insert(0, frames[first_frame_pts])
    return result


def _dvpp_init():
    """ Init dvpp resources. """

    global _INITIALIZED, _INITIALIZED_PID
    if _INITIALIZED and _INITIALIZED_PID != os.getpid():
        raise RuntimeError("Cannot re-initialize Ascend in forked process. To use Ascend with multiprocessing, "
                           "you must use the 'spawn' start method "
                           "via 'mindspore.dataset.config.set_multiprocessing_start_method('spawn')'.")

    if not _INITIALIZED:
        cde.dvpp_sys_init()
        _INITIALIZED = True
        _INITIALIZED_PID = os.getpid()


def _read_video_dvpp(filename, start_pts=0, end_pts=None, pts_unit="pts", output_format="THWC"):
    """ Read video with DVPP. """

    _dvpp_init()

    output_format = output_format.upper()
    if output_format not in ("THWC", "TCHW"):
        raise ValueError(f"output_format should be either 'THWC' or 'TCHW', got {output_format}.")

    info = {}
    audio_frames = []
    audio_timebase = Fraction(0, 1)

    with cde.pyav_open(filename) as container:
        if container.streams.audio:
            audio_timebase = container.streams.audio[0].time_base

        if container.streams.video:
            if container.streams.video[0].name not in ("hevc", "h264"):
                logger.warning(f"This video in {filename} is coding by {container.streams.video[0].name}, "
                               "not supported on DVPP backend and will fall back to run on the FFMPEG."
                               "This may have performance implications.")
                return _read_video_ffmpeg(filename, start_pts, end_pts, pts_unit)

            vframes = _read_from_stream_dvpp(
                filename,
                container,
                start_pts,
                end_pts,
                pts_unit,
                container.streams.video[0],
                {"video": 0})

            video_fps = container.streams.video[0].average_rate
            # guard against potentially corrupted files
            if video_fps is not None:
                info["video_fps"] = float(video_fps)
        else:
            vframes = np.empty(0, dtype=np.uint8)

        if container.streams.audio:
            audio_frames = _read_from_stream_ffmpeg(
                container,
                start_pts,
                end_pts,
                pts_unit,
                container.streams.audio[0],
                {"audio": 0},
            )
            info["audio_fps"] = container.streams.audio[0].rate

    aframes_list = []
    for frame in audio_frames:
        aaa = np.vstack(frame.to_ndarray())
        aframes_list.append(aaa)

    if aframes_list:
        aframes = np.concatenate(aframes_list, 1)
        if pts_unit == "sec" and audio_timebase != 0:
            start_pts = int(math.floor(start_pts * (1 / audio_timebase)))
            if end_pts != float("inf") and audio_timebase != 0:
                end_pts = int(math.ceil(end_pts * (1 / audio_timebase)))
        aframes = _align_audio_frames(aframes, audio_frames, start_pts, end_pts)
    else:
        aframes = np.empty((1, 0), dtype=np.float32)

    if output_format == "THWC" and vframes is not None and vframes.size != 0:
        # [T,C,H,W] --> [T,H,W,C]
        vframes = vframes.transpose(0, 2, 3, 1)

    return vframes, aframes, info


def _read_video_ffmpeg(filename, start_pts=0, end_pts=None, pts_unit="pts"):
    """ Read video with FFMPEG. """

    video_output, audio_output, raw_metadata = cde.read_video(filename, float(start_pts), float(end_pts), pts_unit)

    if video_output is not None:
        video_output = video_output.as_array()
    if audio_output is not None:
        audio_output = audio_output.as_array()
    metadata_output = {}
    for key in raw_metadata:
        if key == "video_fps":
            metadata_output[key] = float(raw_metadata[key])
            continue
        if key == "audio_fps":
            metadata_output[key] = int(raw_metadata[key])
            continue
        metadata_output[key] = raw_metadata[key]
    return video_output, audio_output, metadata_output


def read_video(filename, start_pts=0, end_pts=None, pts_unit="pts"):
    """
    Read the video, audio, and metadata from a video file.

    It supports AVI, H264, H265, MOV, MP4, WMV file formats on CPU, and H264, H265 file formats on Ascend.

    Note:
        This method is executed on CPU by default, but it is also supported to be executed on Ascend by
        setting video backend with `mindspore.dataset.config.set_video_backend("Ascend")` .

    Args:
        filename (str): The path to the video file to be read.
        start_pts (Union[float, Fraction, int], optional): The start presentation timestamp of the video.
            Default: ``0``, read from the beginning.
        end_pts (Union[float, Fraction, int], optional): The end presentation timestamp of the video.
            Default: ``None``, read until the end.
        pts_unit (str, optional): The unit of the timestamps. It can be any of ["pts", "sec"]. Default: ``"pts"``.

    Returns:
        - numpy.ndarray, four-dimensional uint8 data for video. The format is [T, H, W, C]. `T` is
          the number of frames, `H` is the height, `W` is the width, `C` is the number of channels
          for RGB.
        - numpy.ndarray, two-dimensional float data for audio. The format is [C, L]. `C` is the
          number of channels. `L` is the length of the samples in one channel.
        - dict, metadata for the video and audio.
          It contains video_fps data of type float and audio_fps data of type int.

    Raises:
        TypeError: If `filename` is not of type str.
        TypeError: If `start_pts` is not of type Union[float, Fraction, int].
        TypeError: If `end_pts` is not of type Union[float, Fraction, int].
        TypeError: If `pts_unit` is not of type str.
        RuntimeError: If `filename` does not exist, or is not a regular file, or is not a supported
            video file.
        ValueError: If `start_pts` is less than 0.
        ValueError: If `end_pts` is less than `start_pts`.
        ValueError: If `pts_unit` is not in ["pts", "sec"].

    Supported Platforms:
        ``CPU`` ``Ascend``

    Examples:
        >>> import mindspore.dataset.vision as vision
        >>> video_output, audio_output, metadata_output = vision.read_video("/path/to/file")
    """
    if not isinstance(filename, str):
        raise TypeError("Input filename is not of type {0}, but got: {1}.".format(str, type(filename)))
    if not isinstance(start_pts, (float, Fraction, int)):
        raise TypeError("Input start_pts is not of type [{0}, {1}, {2}], but got: {3}".format(float, Fraction, int,
                                                                                              type(start_pts)))
    if start_pts < 0.0:
        err_msg = "Not supported start_pts for " + str(start_pts) + ". The start_pts should be >= 0."
        raise ValueError(err_msg)
    if end_pts is None:
        end_pts = 2147483647.0
    if not isinstance(end_pts, (float, Fraction, int)):
        raise TypeError("Input end_pts is not of type [{0}, {1}, {2}], but got: {3}".format(float, Fraction, int,
                                                                                            type(end_pts)))
    if end_pts < start_pts:
        err_msg = "Not supported end_pts for " + str(end_pts) + ". start_pts = " + str(start_pts) + "."
        err_msg += " The end_pts should be >= start_pts."
        raise ValueError(err_msg)
    if not isinstance(pts_unit, str):
        raise TypeError("Input pts_unit is not of type {0}, but got: {1}.".format(str, type(pts_unit)))
    if pts_unit not in ["pts", "sec"]:
        raise ValueError("Not supported pts_unit for " + pts_unit)

    filepath = os.path.realpath(filename)

    if not os.path.exists(filepath):
        raise ValueError("Invalid file path, " + filename + " does not exist.")

    if not os.path.isfile(filepath):
        raise ValueError("Invalid file path, " + filename + " is not a regular file.")

    if get_video_backend() == "Ascend":
        return _read_video_dvpp(filename, start_pts, end_pts, pts_unit)
    return _read_video_ffmpeg(filename, start_pts, end_pts, pts_unit)


class VideoDecoder:
    """
    A decoder for single video streams, capable of parsing metadata and extracting frames
    from H264/H265-encoded content.

    Args:
        source (str): The path to the video file.

    Raises:
        TypeError: If `source` is not string.
        ValueError: If `source` does not exist or permission is denied.

    Examples:
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>>
        >>> ds.config.set_video_backend("Ascend")
        >>> reader = vision.VideoDecoder(source="/path/to/filename")
    """
    def __init__(self, source):
        check_file(source)
        self.source = source
        self._metadata = self.metadata

    def get_frames_at(self, indices):
        """
        Retrieves the frames at the specified indices.

        Args:
            indices (list[int]): List of frame indices to acquire.

        Returns:
            numpy.ndarray, four-dimensional uint8 data for video. The format is [T, H, W, C].
            `T` is the number of frames, `H` is the height, `W` is the width, `C` is the number of
            channels for RGB.

        Raises:
            TypeError: If `indices` is not of type list.
            TypeError: If an element in `indices` is not of type int.
            ValueError: If an element in `indices` is not in the range [0, total frames).

        Supported Platforms:
            ``Ascend``

        Examples:
            >>> import mindspore.dataset as ds
            >>> import mindspore.dataset.vision as vision
            >>>
            >>> ds.config.set_video_backend("Ascend")
            >>> reader = vision.VideoDecoder(source="/path/to/filename")
            >>> output_frames = reader.get_frames_at([0, 1, 2, 3])
        """
        if get_video_backend() != "Ascend":
            raise RuntimeError("Method get_frames_at is only supported on Ascend platform.")
        type_check(indices, (list,), "indices")
        type_check_list(indices, (int,), "indices")

        _dvpp_init()

        if indices == []:
            return np.empty(0, dtype=np.uint8)
        for i, frame_index in enumerate(indices):
            check_value(frame_index, [0, self._metadata["num_frames"]], "Invalid frame index[{0}]={1}".format(
                i, indices[i]), right_open_interval=True)
        filepath = os.path.realpath(self.source)

        with cde.pyav_open(filepath) as container:
            if container.streams.video:
                if container.streams.video[0].name in ("hevc", "h264"):
                    vframes = self._read_from_stream_dvpp_frames(
                        filepath,
                        container,
                        0,
                        float("inf"),
                        container.streams.video[0],
                        {"video": 0},
                        indices,
                    )
                else:
                    raise RuntimeError(f"This video in {filepath} is coding by {container.streams.video[0].name}, "
                                       "not supported on DVPP backend.")
            else:
                vframes = np.empty(0, dtype=np.uint8)

        if vframes is not None and vframes.size != 0:
            # [T,C,H,W] --> [T,H,W,C]
            vframes = vframes.transpose(0, 2, 3, 1)

        return vframes

    @property
    def metadata(self):
        """
        Getting metadata of the video stream.

        Returns:
            dict, information about the metadata.

        Examples:
            >>> import mindspore.dataset as ds
            >>> import mindspore.dataset.vision as vision
            >>>
            >>> ds.config.set_video_backend("Ascend")
            >>> reader = vision.VideoDecoder(source="/path/to/filename")
            >>> metadata = reader.metadata
        """
        metadata = {}
        filepath = os.path.realpath(self.source)
        with cde.pyav_open(filepath) as container:
            stream = container.streams.video[0]
            metadata["width"] = stream.width
            metadata["height"] = stream.height
            metadata["duration_seconds"] = round(float(stream.duration * stream.time_base), 6)
            metadata["num_frames"] = stream.frames
            metadata["average_fps"] = float(stream.average_rate)
        return metadata

    def _read_from_stream_dvpp_frames(self, filename, container, start_offset, end_offset,
                                      stream, stream_name, indices):
        """ Read video stream with DVPP. """
        if not stream.type == "video":
            raise RuntimeError("_read_from_stream_dvpp_frames only handle video type")
        max_buffer_size = 5
        # DivX-style packed B-frames can have out-of-order pts (2 frames in a single pkt)
        # so need to buffer some extra frames to sort everything properly
        should_buffer = _check_buffer(stream.codec_context.extradata)

        seek_offset = start_offset
        # some files don't seek to the right location, so better be safe here
        seek_offset = max(seek_offset - 1, 0)

        if should_buffer:
            seek_offset = max(seek_offset - max_buffer_size, 0)
        # init frames before seek
        frames, pts_per_frame = _get_frame_by_cv(filename, container, stream)
        container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
        decode_params = DecodeParams(container, start_offset, end_offset, stream)
        frames = self._decode_video_dvpp_frames(decode_params, frames, pts_per_frame, indices)

        if frames is None:
            logger.warning(f"_decode_video_dvpp failed: {filename}")

        return frames

    def _get_key_frames_and_first_pts(self, container, stream, pts_per_frame):
        """ Get key frames and first pts. """
        key_list = []
        first_pts = None
        for packet in container.demux(stream):
            if packet.pts is not None:
                if packet.is_keyframe:
                    key_list.append(int(packet.pts * stream.time_base * stream.average_rate))
                    first_pts = packet.pts % pts_per_frame
        return key_list, first_pts

    def _compute_ret_tensor(self, container, stream, frames_count, frames_in_group, frames, pts_per_frame,
                            ret_tensor, target_frame_positions, start_frame, chn):
        """ Compute ret_tensor. """
        for packet in container.demux(stream):
            if packet.pts is not None:
                if frames_count == len(frames_in_group):
                    break
                frame = frames[packet.pts].frame
                input_tensor = cde.DeviceBuffer.from_numpy(frame)
                if packet.pts // pts_per_frame in frames_in_group:
                    display = True
                    frames_count += 1
                    output_tensor = \
                        ret_tensor[target_frame_positions[frames[packet.pts].positions - start_frame][0]]
                else:
                    display = False
                    output_tensor = cde.DeviceBuffer([])
                cde.decode_video_send_stream(chn, input_tensor, 69, display, output_tensor)

    def _decode_video_dvpp_frames(self, decode_params, frames, pts_per_frame, indices):
        """ Send frames to Ascend and using DVPP to decode. """

        container = decode_params.container
        start_offset = decode_params.start_offset
        end_offset = decode_params.end_offset
        stream = decode_params.stream

        codecs_type = stream.name
        hi_pt_h264 = 96
        hi_pt_h265 = 265
        if codecs_type == "h264":
            codec_id = hi_pt_h264
        elif codecs_type == "hevc":
            codec_id = hi_pt_h265
        else:
            raise ValueError(f"The video codecs_type should be either 'h264' or 'hevc', got {codecs_type}.")

        # if start_offset is between 2 frames, get one more previous frame
        start_offset = int(start_offset / pts_per_frame) * pts_per_frame

        frame_width = stream.width
        frame_height = stream.height

        # update end_offset_real
        end_offset_real = end_offset
        end_offset_real = min(end_offset_real, len(frames) * pts_per_frame)
        start_frame = math.ceil(start_offset / pts_per_frame)
        total_frame = math.ceil((end_offset_real - start_offset) / pts_per_frame)

        if end_offset_real < start_offset or total_frame == 0:
            return np.empty(0, dtype=np.uint8)
        target_frame_list = list(set(indices))
        target_frame_list.sort()
        target_frame_positions = {}
        for index, value in enumerate(target_frame_list):
            target_frame_positions.setdefault(value, []).append(index)
        ret_tensor = cde.DeviceBuffer([len(target_frame_list), 3, frame_height, frame_width])

        # decode from dvpp
        chn = cde.decode_video_create_chn(codec_id)
        cde.decode_video_start_get_frame(chn, len(target_frame_list))

        groups = {}
        key_list, first_pts = self._get_key_frames_and_first_pts(container, stream, pts_per_frame)

        for frame in target_frame_list:
            keyframe = max(k for k in key_list if k <= frame)
            groups.setdefault(keyframe, []).append(frame)
        container.seek(0, any_frame=False, backward=True, stream=stream)
        average_rate = stream.average_rate
        time_base = stream.time_base

        for keyframe, frames_in_group in groups.items():
            frames_count = 0
            timestamp = keyframe / average_rate
            seek_target = int(timestamp / time_base + first_pts)
            container.seek(seek_target, any_frame=False, backward=True, stream=stream)
            self._compute_ret_tensor(container, stream, frames_count, frames_in_group, frames,
                                     pts_per_frame, ret_tensor, target_frame_positions, start_frame, chn)

        # ret_tensor is ordered by pts len(target_frame_list)
        ret_tensor_dvpp = cde.decode_video_stop_get_frame(chn, len(target_frame_list))

        # if ret_tensor_dvpp empty, means ret_tensor already filled
        if ret_tensor_dvpp.size() != 0:
            ret_tensor = ret_tensor_dvpp

        ret_numpy = ret_tensor.numpy()

        cde.decode_video_destroy_chnl(chn)

        if indices != target_frame_list:
            mapping = {val: ret_numpy[index] for index, val in enumerate(target_frame_list)}
            ret_numpy = np.stack([mapping[val] for val in indices])
        return ret_numpy


def read_video_timestamps(filename, pts_unit="pts"):
    """
    Read the timestamps and frames per second of a video file.
    It supports AVI, H264, H265, MOV, MP4, WMV files.

    Args:
        filename (str): The path to the video file to be read.
        pts_unit (str, optional): The unit of the timestamps. It can be any of ["pts", "sec"].
            Default: ``"pts"``.

    Returns:
        - list. When `pts_unit` is set to ``"pts"``, list[int] is returned; when `pts_unit` is set to
          ``"sec"``, list[float] is returned.
        - float, the frames per second of the video file.

    Raises:
        TypeError: If `filename` is not of type str.
        TypeError: If `pts_unit` is not of type str.
        RuntimeError: If `filename` does not exist, or is not a regular file, or is not a supported
            video file.
        RuntimeError: If `pts_unit` is not in ["pts", "sec"].

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import mindspore.dataset.vision as vision
        >>> video_timestamps, video_fps = vision.read_video_timestamps("/path/to/file")
    """
    if not isinstance(filename, str):
        raise TypeError("Input filename is not of type {0}, but got: {1}.".format(str, type(filename)))
    if not isinstance(pts_unit, str):
        raise TypeError("Input pts_unit is not of type {0}, but got: {1}.".format(str, type(pts_unit)))

    video_pts, video_fps, time_base = cde.read_video_timestamps(filename, pts_unit)

    if video_pts == []:
        return video_pts, None
    if pts_unit == "pts":
        return video_pts, video_fps
    return [x * time_base for x in video_pts], video_fps


def write_file(filename, data):
    """
    Write the one-dimensional uint8 data into a file using binary mode.

    Args:
        filename (str): The path to the file to be written.
        data (Union[numpy.ndarray, mindspore.Tensor]): The one-dimensional uint8 data to be written.

    Raises:
        TypeError: If `filename` is not of type str.
        TypeError: If `data` is not of type numpy.ndarray or mindspore.Tensor.
        RuntimeError: If `filename` is not a regular file.
        RuntimeError: If the data type of `data` is not uint8.
        RuntimeError: If `data` is not a one-dimensional array.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import mindspore.dataset.vision as vision
        >>> import numpy as np
        >>> # Generate a random data with 1024 bytes
        >>> data = np.random.randint(256, size=(1024), dtype=np.uint8)
        >>> vision.write_file("/path/to/file", data)
    """
    if not isinstance(filename, str):
        raise TypeError("Input filename is not of type {0}, but got: {1}.".format(str, type(filename)))
    if isinstance(data, np.ndarray):
        return cde.write_file(filename, cde.Tensor(data))
    if isinstance(data, mindspore.Tensor):
        return cde.write_file(filename, cde.Tensor(data.asnumpy()))
    raise TypeError("Input data is not of type {0} or {1}, but got: {2}.".format(np.ndarray,
                                                                                 mindspore.Tensor, type(data)))


def write_jpeg(filename, image, quality=75):
    """
    Write the image data into a JPEG file.

    Args:
        filename (str): The path to the file to be written.
        image (Union[numpy.ndarray, mindspore.Tensor]): The image data to be written.
        quality (int, optional): Quality of the resulting JPEG file, in the range [1, 100].
            Default: ``75``.

    Raises:
        TypeError: If `filename` is not of type str.
        TypeError: If `image` is not of type numpy.ndarray or mindspore.Tensor.
        TypeError: If `quality` is not of type int.
        RuntimeError: If `filename` does not exist or is not a regular file.
        RuntimeError: If the data type of `image` is not uint8.
        RuntimeError: If the shape of `image` is not <H, W> or <H, W, 1> or <H, W, 3>.
        RuntimeError: If `quality` is less than 1 or greater than 100.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import mindspore.dataset.vision as vision
        >>> import numpy as np
        >>> # Generate a random image with height=120, width=340, channels=3
        >>> image = np.random.randint(256, size=(120, 340, 3), dtype=np.uint8)
        >>> vision.write_jpeg("/path/to/file", image)
    """
    if not isinstance(filename, str):
        raise TypeError("Input filename is not of type {0}, but got: {1}.".format(str, type(filename)))
    if not isinstance(quality, int):
        raise TypeError("Input quality is not of type {0}, but got: {1}.".format(int, type(quality)))
    if isinstance(image, np.ndarray):
        return cde.write_jpeg(filename, cde.Tensor(image), quality)
    if isinstance(image, mindspore.Tensor):
        return cde.write_jpeg(filename, cde.Tensor(image.asnumpy()), quality)
    raise TypeError("Input image is not of type {0} or {1}, but got: {2}.".format(np.ndarray,
                                                                                  mindspore.Tensor, type(image)))


def write_png(filename, image, compression_level=6):
    """
    Write the image data into a PNG file.

    Args:
        filename (str): The path to the file to be written.
        image (Union[numpy.ndarray, mindspore.Tensor]): The image data to be written.
        compression_level (int, optional): Compression level for the resulting PNG file, in the range
            [0, 9]. Default: ``6``.

    Raises:
        TypeError: If `filename` is not of type str.
        TypeError: If `image` is not of type numpy.ndarray or mindspore.Tensor.
        TypeError: If `compression_level` is not of type int.
        RuntimeError: If `filename` does not exist or is not a regular file.
        RuntimeError: If the data type of `image` is not uint8.
        RuntimeError: If the shape of `image` is not <H, W> or <H, W, 1> or <H, W, 3>.
        RuntimeError: If `compression_level` is less than 0 or greater than 9.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import mindspore.dataset.vision as vision
        >>> import numpy as np
        >>> # Generate a random image with height=120, width=340, channels=3
        >>> image = np.random.randint(256, size=(120, 340, 3), dtype=np.uint8)
        >>> vision.write_png("/path/to/file", image)
    """
    if not isinstance(filename, str):
        raise TypeError("Input filename is not of type {0}, but got: {1}.".format(str, type(filename)))
    if not isinstance(compression_level, int):
        raise TypeError("Input compression_level is not of type {0}, but got: {1}.".format(int,
                                                                                           type(compression_level)))
    if isinstance(image, np.ndarray):
        return cde.write_png(filename, cde.Tensor(image), compression_level)
    if isinstance(image, mindspore.Tensor):
        return cde.write_png(filename, cde.Tensor(image.asnumpy()), compression_level)
    raise TypeError("The input image is not of type {0} or {1}, but got: {2}.".format(np.ndarray,
                                                                                      mindspore.Tensor, type(image)))
