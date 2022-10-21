# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
from enum import Enum, IntEnum
import numbers

import numpy as np
from PIL import Image

import mindspore
import mindspore._c_dataengine as cde


class AutoAugmentPolicy(str, Enum):
    """
    AutoAugment policy for different datasets.

    Possible enumeration values are: AutoAugmentPolicy.IMAGENET, AutoAugmentPolicy.CIFAR10,
    AutoAugmentPolicy.SVHN.

    Each policy contains 25 pairs of augmentation operations. When using AutoAugment, each image is randomly
    transformed with one of these operation pairs. Each pair has 2 different operations. The following shows
    all of these augmentation operations, including operation names with their probabilities and random params.

    - AutoAugmentPolicy.IMAGENET: dataset auto augment policy for ImageNet.

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

    - AutoAugmentPolicy.CIFAR10: dataset auto augment policy for Cifar10.

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

    - AutoAugmentPolicy.SVHN: dataset auto augment policy for SVHN.

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

        return c_values.get(policy)


class Border(str, Enum):
    """
    Padding Mode, Border Type.

    Possible enumeration values are: Border.CONSTANT, Border.EDGE, Border.REFLECT, Border.SYMMETRIC.

    - Border.CONSTANT: means it fills the border with constant values.
    - Border.EDGE: means it pads with the last value on the edge.
    - Border.REFLECT: means it reflects the values on the edge omitting the last value of edge.
      For example, padding [1,2,3,4] with 2 elements on both sides will result in [3,2,1,2,3,4,3,2].
    - Border.SYMMETRIC: means it reflects the values on the edge repeating the last value of edge.
      For example, padding [1,2,3,4] with 2 elements on both sides will result in [2,1,1,2,3,4,4,3].

    Note:
        This class derived from class str to support json serializable.
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
        return python_values.get(border_type)

    @staticmethod
    def to_c_type(border_type):
        """
        Function to return C type for Border Type.
        """
        c_values = {Border.CONSTANT: cde.BorderType.DE_BORDER_CONSTANT,
                    Border.EDGE: cde.BorderType.DE_BORDER_EDGE,
                    Border.REFLECT: cde.BorderType.DE_BORDER_REFLECT,
                    Border.SYMMETRIC: cde.BorderType.DE_BORDER_SYMMETRIC}

        return c_values.get(border_type)


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

        return c_values.get(mode)


class ImageBatchFormat(IntEnum):
    """
    Data Format of images after batch operation.

    Possible enumeration values are: ImageBatchFormat.NHWC, ImageBatchFormat.NCHW.

    - ImageBatchFormat.NHWC: in orders like, batch N, height H, width W, channels C to store the data.
    - ImageBatchFormat.NCHW: in orders like, batch N, channels C, height H, width W to store the data.
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

        return c_values.get(image_batch_format)


class Inter(IntEnum):
    """
    Interpolation Modes.

    Possible enumeration values are: Inter.NEAREST, Inter.ANTIALIAS, Inter.LINEAR, Inter.BILINEAR, Inter.CUBIC,
    Inter.BICUBIC, Inter.AREA, Inter.PILCUBIC.

    - Inter.NEAREST: means interpolation method is nearest-neighbor interpolation.
    - Inter.ANTIALIAS: means the interpolation method is antialias interpolation.
    - Inter.LINEAR: means interpolation method is bilinear interpolation, here is the same as Inter.BILINEAR.
    - Inter.BILINEAR: means interpolation method is bilinear interpolation.
    - Inter.CUBIC: means the interpolation method is bicubic interpolation, here is the same as Inter.BICUBIC.
    - Inter.BICUBIC: means the interpolation method is bicubic interpolation.
    - Inter.AREA: means interpolation method is pixel area interpolation.
    - Inter.PILCUBIC: means interpolation method is bicubic interpolation like implemented in pillow, input
      should be in 3 channels format.
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
        if Image.__version__ >= "9.1.0":
            python_values = {Inter.NEAREST: Image.Resampling.NEAREST,
                             Inter.ANTIALIAS: Image.Resampling.LANCZOS,
                             Inter.LINEAR: Image.Resampling.BILINEAR,
                             Inter.CUBIC: Image.Resampling.BICUBIC}
        else:
            python_values = {Inter.NEAREST: Image.NEAREST,
                             Inter.ANTIALIAS: Image.ANTIALIAS,
                             Inter.LINEAR: Image.LINEAR,
                             Inter.CUBIC: Image.CUBIC}
        return python_values.get(inter_type)

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

        return c_values.get(inter_type)


class SliceMode(IntEnum):
    """
    Mode to Slice Tensor into multiple parts.

    Possible enumeration values are: SliceMode.PAD, SliceMode.DROP.

    - SliceMode.PAD: pad some pixels before slice the Tensor if needed.
    - SliceMode.DROP: drop remainder pixels before slice the Tensor if needed.
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

        return c_values.get(mode)


def encode_jpeg(image, quality=75):
    """
    Encode the input image as JPEG data.

    Args:
        image (Union[numpy.ndarray, mindspore.Tensor]): The image to be encoded.
        quality (int, optional): Quality of the resulting JPEG data, from 1 to 100. Default: 75.

    Returns:
        numpy.ndarray, one dimension uint8 data.

    Raises:
        TypeError: If `image` is not of type numpy.ndarray or mindspore.Tensor.
        TypeError: If `quality` is not of type int.
        RuntimeError: If the data type of `image` is not uint8.
        RuntimeError: If the shape of `image` is not <H, W> or <H, W, 1> or <H, W, 3>.
        RuntimeError: If `quality` is less than 1 or greater than 100.

    Examples:
        >>> import numpy as np
        >>> from mindspore.dataset import vision
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


def get_image_num_channels(image):
    """
    Get the number of input image channels.

    Args:
        image (Union[numpy.ndarray, PIL.Image.Image]): Image to get the number of channels.

    Returns:
        int, the number of input image channels.

    Examples:
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
        image (Union[numpy.ndarray, PIL.Image.Image]): The image to get size.

    Returns:
        list[int, int], the image size.

    Examples:
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
