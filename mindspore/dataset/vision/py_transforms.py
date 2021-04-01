# Copyright 2019 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
The module vision.py_transforms is implemented based on Python PIL.
This module provides many kinds of image augmentations. It also provides
transferring methods between PIL image and NumPy array. For users who prefer
Python PIL in image learning tasks, this module is a good tool to process images.
Users can also self-define their own augmentations with Python PIL.
"""
import numbers
import random

import numpy as np
from PIL import Image

from . import py_transforms_util as util
from .c_transforms import parse_padding
from .validators import check_prob, check_crop, check_resize_interpolation, check_random_resize_crop, \
    check_normalize_py, check_normalizepad_py, check_random_crop, check_random_color_adjust, check_random_rotation, \
    check_ten_crop, check_num_channels, check_pad, \
    check_random_perspective, check_random_erasing, check_cutout, check_linear_transform, check_random_affine, \
    check_mix_up, check_positive_degrees, check_uniform_augment_py, check_auto_contrast
from .utils import Inter, Border
from .py_transforms_util import is_pil

DE_PY_INTER_MODE = {Inter.NEAREST: Image.NEAREST,
                    Inter.ANTIALIAS: Image.ANTIALIAS,
                    Inter.LINEAR: Image.LINEAR,
                    Inter.CUBIC: Image.CUBIC}

DE_PY_BORDER_TYPE = {Border.CONSTANT: 'constant',
                     Border.EDGE: 'edge',
                     Border.REFLECT: 'reflect',
                     Border.SYMMETRIC: 'symmetric'}


def not_random(function):
    function.random = False
    return function


class ToTensor:
    """
    Convert the input NumPy image array or PIL image of shape (H, W, C) to a NumPy ndarray of shape (C, H, W).

    Note:
        The values in the input arrays are rescaled from [0, 255] to [0.0, 1.0].
        The type is cast to output_type (default NumPy float32).
        The number of channels remains the same.

    Args:
        output_type (NumPy datatype, optional): The datatype of the NumPy output (default=np.float32).

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> # create a list of transformations to be applied to the "image" column of each data row
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                           py_vision.RandomHorizontalFlip(0.5),
        ...                           py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    def __init__(self, output_type=np.float32):
        self.output_type = output_type
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): PIL image to be converted to numpy.ndarray.

        Returns:
            img (numpy.ndarray), Converted image.
        """
        return util.to_tensor(img, self.output_type)


class ToType:
    """
    Convert the input NumPy image array to desired NumPy dtype.

    Args:
        output_type (NumPy datatype): The datatype of the NumPy output, e.g. numpy.float32.

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> import numpy as np
        >>> transforms_list =Compose([py_vision.Decode(),
        ...                           py_vision.RandomHorizontalFlip(0.5),
        ...                           py_vision.ToTensor(),
        ...                           py_vision.ToType(np.float32)])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    def __init__(self, output_type):
        self.output_type = output_type
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            NumPy object : NumPy object to be type swapped.

        Returns:
            img (numpy.ndarray), Converted image.
        """
        return util.to_type(img, self.output_type)


class HWC2CHW:
    """
    Transpose a NumPy image array; shape (H, W, C) to shape (C, H, W).

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.HWC2CHW()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    def __init__(self):
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (numpy.ndarray): Image array, of shape (H, W, C), to have channels swapped.

        Returns:
            img (numpy.ndarray), Image array, of shape (C, H, W), with channels swapped.
        """
        return util.hwc_to_chw(img)


class ToPIL:
    """
    Convert the input decoded NumPy image array of RGB mode to a PIL image of RGB mode.

    Examples:
        >>> # data is already decoded, but not in PIL image format
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.ToPIL(),
        ...                            py_vision.RandomHorizontalFlip(0.5),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    def __init__(self):
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (numpy.ndarray): Decoded image array, of RGB mode, to be converted to PIL image.

        Returns:
            img (PIL image), Image converted to PIL image of RGB mode.
        """
        return util.to_pil(img)


class Decode:
    """
    Decode the input image to PIL image format in RGB mode.

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomHorizontalFlip(0.5),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    def __init__(self):
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (Bytes-like Objects):Image to be decoded.

        Returns:
            img (PIL image), Decoded image in RGB mode.
        """
        return util.decode(img)


class Normalize:
    """
    Normalize the input NumPy image array of shape (C, H, W) with the given mean and standard deviation.

    The values of the array need to be in the range (0.0, 1.0].

    Args:
        mean (sequence): List or tuple of mean values for each channel, with respect to channel order.
            The mean values must be in the range [0.0, 1.0].
        std (sequence): List or tuple of standard deviations for each channel, w.r.t. channel order.
            The standard deviation values must be in the range (0.0, 1.0].

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomHorizontalFlip(0.5),
        ...                            py_vision.ToTensor(),
        ...                            py_vision.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262))])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_normalize_py
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (numpy.ndarray): Image array to be normalized.

        Returns:
            img (numpy.ndarray), Normalized Image array.
        """
        return util.normalize(img, self.mean, self.std)


class NormalizePad:
    """
    Normalize the input NumPy image array of shape (C, H, W) with the given mean and standard deviation
        then pad an extra channel with value zero.

    The values of the array need to be in the range (0.0, 1.0].

    Args:
        mean (sequence): List or tuple of mean values for each channel, with respect to channel order.
            The mean values must be in the range (0.0, 1.0].
        std (sequence): List or tuple of standard deviations for each channel, w.r.t. channel order.
            The standard deviation values must be in the range (0.0, 1.0].
        dtype (str): Set the output data type of image (default is "float32").

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomHorizontalFlip(0.5),
        ...                            py_vision.ToTensor(),
        ...                            py_vision.NormalizePad((0.491, 0.482, 0.447), (0.247, 0.243, 0.262), "float32")])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_normalizepad_py
    def __init__(self, mean, std, dtype="float32"):
        self.mean = mean
        self.std = std
        self.dtype = dtype
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (numpy.ndarray): Image array to be normalizepad.

        Returns:
            img (numpy.ndarray), NormalizePaded Image array.
        """
        return util.normalize(img, self.mean, self.std, pad_channel=True, dtype=self.dtype)


class RandomCrop:
    """
    Crop the input PIL image at a random location.

    Args:
        size (Union[int, sequence]): The output size of the cropped image.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).
        padding (Union[int, sequence], optional): The number of pixels to pad the image (default=None).
            If padding is not None, first pad image with padding values.
            If a single number is provided, pad all borders with this value.
            If a tuple or list of 2 values are provided, pad the (left and top)
            with the first value and (right and bottom) with the second value.
            If 4 values are provided as a list or tuple,
            pad the left, top, right and bottom respectively.
        pad_if_needed (bool, optional): Pad the image if either side is smaller than
            the given output size (default=False).
        fill_value (int or tuple, optional): filling value (default=0).
            The pixel intensity of the borders if the padding_mode is Border.CONSTANT.
            If it is a 3-tuple, it is used to fill R, G, B channels respectively.
        padding_mode (str, optional): The method of padding (default=Border.CONSTANT). It can be any of
            [Border.CONSTANT, Border.EDGE, Border.REFLECT, Border.SYMMETRIC].

            - Border.CONSTANT, means it fills the border with constant values.

            - Border.EDGE, means it pads with the last value on the edge.

            - Border.REFLECT, means it reflects the values on the edge omitting the last
              value of edge.

            - Border.SYMMETRIC, means it reflects the values on the edge repeating the last
              value of edge.

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomCrop(224),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_random_crop
    def __init__(self, size, padding=None, pad_if_needed=False, fill_value=0, padding_mode=Border.CONSTANT):
        if padding is None:
            padding = (0, 0, 0, 0)
        else:
            padding = parse_padding(padding)
        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill_value = fill_value
        self.padding_mode = DE_PY_BORDER_TYPE[padding_mode]

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to be randomly cropped.

        Returns:
            PIL image, Cropped image.
        """
        return util.random_crop(img, self.size, self.padding, self.pad_if_needed,
                                self.fill_value, self.padding_mode)


class RandomHorizontalFlip:
    """
    Randomly flip the input image horizontally with a given probability.

    Args:
        prob (float, optional): Probability of the image being flipped (default=0.5).

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomHorizontalFlip(0.5),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_prob
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to be flipped horizontally.

        Returns:
            img (PIL image), Randomly flipped image.
        """
        return util.random_horizontal_flip(img, self.prob)


class RandomVerticalFlip:
    """
    Randomly flip the input image vertically with a given probability.

    Args:
        prob (float, optional): Probability of the image being flipped (default=0.5).

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomVerticalFlip(0.5),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_prob
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to be flipped vertically.

        Returns:
            img (PIL image), Randomly flipped image.
        """
        return util.random_vertical_flip(img, self.prob)


class Resize:
    """
    Resize the input PIL image to the given size.

    Args:
        size (Union[int, sequence]): The output size of the resized image.
            If size is an integer, the smaller edge of the image will be resized to this value with
            the same image aspect ratio.
            If size is a sequence of length 2, it should be (height, width).
        interpolation (Inter mode, optional): Image interpolation mode (default=Inter.BILINEAR).
            It can be any of [Inter.NEAREST, Inter.ANTIALIAS, Inter.BILINEAR, Inter.BICUBIC].

            - Inter.NEAREST, means the interpolation method is nearest-neighbor interpolation.

            - Inter.ANTIALIAS, means the interpolation method is antialias interpolation.

            - Inter.BILINEAR, means the interpolation method is bilinear interpolation.

            - Inter.BICUBIC, means the interpolation method is bicubic interpolation.

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.Resize(256),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_resize_interpolation
    def __init__(self, size, interpolation=Inter.BILINEAR):
        self.size = size
        self.interpolation = DE_PY_INTER_MODE[interpolation]
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to be resized.

        Returns:
            img (PIL image), Resize image.
        """
        return util.resize(img, self.size, self.interpolation)


class RandomResizedCrop:
    """
    Extract crop from the input image and resize it to a random size and aspect ratio.

    Args:
        size (Union[int, sequence]): The size of the output image.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).
        scale (tuple, optional): Range (min, max) of respective size of the original size
            to be cropped (default=(0.08, 1.0)).
        ratio (tuple, optional): Range (min, max) of aspect ratio to be cropped (default=(3. / 4., 4. / 3.)).
        interpolation (Inter mode, optional): Image interpolation mode (default=Inter.BILINEAR).
            It can be any of [Inter.NEAREST, Inter.ANTIALIAS, Inter.BILINEAR, Inter.BICUBIC].

            - Inter.NEAREST, means the interpolation method is nearest-neighbor interpolation.

            - Inter.ANTIALIAS, means the interpolation method is antialias interpolation.

            - Inter.BILINEAR, means the interpolation method is bilinear interpolation.

            - Inter.BICUBIC, means the interpolation method is bicubic interpolation.

        max_attempts (int, optional): The maximum number of attempts to propose a valid
            crop area (default=10). If exceeded, fall back to use center crop instead.

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomResizedCrop(224),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_random_resize_crop
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation=Inter.BILINEAR, max_attempts=10):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = DE_PY_INTER_MODE[interpolation]
        self.max_attempts = max_attempts

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to be randomly cropped and resized.

        Returns:
            img (PIL image), Randomly cropped and resized image.
        """
        return util.random_resize_crop(img, self.size, self.scale, self.ratio,
                                       self.interpolation, self.max_attempts)


class CenterCrop:
    """
    Crop the central reigion of the input PIL image to the given size.

    Args:
        size (Union[int, sequence]): The output size of the cropped image.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.CenterCrop(64),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_crop
    def __init__(self, size):
        self.size = size
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to be center cropped.

        Returns:
            img (PIL image), Cropped image.
        """
        return util.center_crop(img, self.size)


class RandomColorAdjust:
    """
    Perform a random brightness, contrast, saturation, and hue adjustment on the input PIL image.

    Args:
        brightness (Union[float, tuple], optional): Brightness adjustment factor (default=(1, 1)). Cannot be negative.
            If it is a float, the factor is uniformly chosen from the range [max(0, 1-brightness), 1+brightness].
            If it is a sequence, it should be [min, max] for the range.
        contrast (Union[float, tuple], optional): Contrast adjustment factor (default=(1, 1)). Cannot be negative.
            If it is a float, the factor is uniformly chosen from the range [max(0, 1-contrast), 1+contrast].
            If it is a sequence, it should be [min, max] for the range.
        saturation (Union[float, tuple], optional): Saturation adjustment factor (default=(1, 1)). Cannot be negative.
            If it is a float, the factor is uniformly chosen from the range [max(0, 1-saturation), 1+saturation].
            If it is a sequence, it should be [min, max] for the range.
        hue (Union[float, tuple], optional): Hue adjustment factor (default=(0, 0)).
            If it is a float, the range will be [-hue, hue]. Value should be 0 <= hue <= 0.5.
            If it is a sequence, it should be [min, max] where -0.5 <= min <= max <= 0.5.

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomColorAdjust(0.4, 0.4, 0.4, 0.1),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_random_color_adjust
    def __init__(self, brightness=(1, 1), contrast=(1, 1), saturation=(1, 1), hue=(0, 0)):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to have its color adjusted randomly.

        Returns:
            img (PIL image), Image after random adjustment of its color.
        """
        return util.random_color_adjust(img, self.brightness, self.contrast, self.saturation, self.hue)


class RandomRotation:
    """
    Rotate the input PIL image by a random angle.

    Note:
        See https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.rotate.

    Args:
        degrees (Union[int, float, sequence]): Range of random rotation degrees.
            If degrees is a number, the range will be converted to (-degrees, degrees).
            If degrees is a sequence, it should be (min, max).
        resample (Inter mode, optional): An optional resampling filter (default=Inter.NEAREST).
            If omitted, or if the image has mode "1" or "P", it is set to be Inter.NEAREST.
            It can be any of [Inter.NEAREST, Inter.ANTIALIAS, Inter.BILINEAR, Inter.BICUBIC].

            - Inter.NEAREST, means the resampling method is nearest-neighbor interpolation.

            - Inter.ANTIALIAS, means the resampling method is antialias interpolation.

            - Inter.BILINEAR, means the resampling method is bilinear interpolation.

            - Inter.BICUBIC, means the resampling method is bicubic interpolation.

        expand (bool, optional):  Optional expansion flag (default=False). If set to True, expand the output
            image to make it large enough to hold the entire rotated image.
            If set to False or omitted, make the output image the same size as the input.
            Note that the expand flag assumes rotation around the center and no translation.
        center (tuple, optional): Optional center of rotation (a 2-tuple) (default=None).
            Origin is the top left corner. Default None sets to the center of the image.
        fill_value (int or tuple, optional): Optional fill color for the area outside the rotated
            image (default=0).
            If it is a 3-tuple, it is used for R, G, B channels respectively.
            If it is an integer, it is used for all RGB channels. Default is 0.

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomRotation(30),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_random_rotation
    def __init__(self, degrees, resample=Inter.NEAREST, expand=False, center=None, fill_value=0):
        self.degrees = degrees
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
        return util.random_rotation(img, self.degrees, self.resample, self.expand, self.center, self.fill_value)


class FiveCrop:
    """
    Generate 5 cropped images (one central image and four corners images).

    Args:
        size (int or sequence): The output size of the crop.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.FiveCrop(size=200),
        ...                            # 4D stack of 5 images
        ...                            lambda *images: numpy.stack([py_vision.ToTensor()(image) for image in images])])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_crop
    def __init__(self, size):
        self.size = size
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): PIL image to be cropped.

        Returns:
            img_tuple (tuple), a tuple of 5 PIL images
                (top_left, top_right, bottom_left, bottom_right, center).
        """
        return util.five_crop(img, self.size)


class TenCrop:
    """
    Generate 10 cropped images (first 5 images from FiveCrop, second 5 images from their flipped version
    as per input flag to flip vertically or horizontally).

    Args:
        size (Union[int, sequence]): The output size of the crop.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).
        use_vertical_flip (bool, optional): Flip the image vertically instead of horizontally
            if set to True (default=False).

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.TenCrop(size=200),
        ...                            # 4D stack of 10 images
        ...                            lambda *images: numpy.stack([py_vision.ToTensor()(image) for image in images])])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_ten_crop
    def __init__(self, size, use_vertical_flip=False):
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.use_vertical_flip = use_vertical_flip
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): PIL image to be cropped.

        Returns:
            img_tuple (tuple), a tuple of 10 PIL images
                (top_left, top_right, bottom_left, bottom_right, center) of original image +
                (top_left, top_right, bottom_left, bottom_right, center) of flipped image.
        """
        return util.ten_crop(img, self.size, self.use_vertical_flip)


class Grayscale:
    """
    Convert the input PIL image to grayscale image.

    Args:
        num_output_channels (int): Number of channels of the output grayscale image (1 or 3).
            Default is 1. If set to 3, the returned image has 3 identical RGB channels.

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.Grayscale(3),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_num_channels
    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): PIL image to be converted to grayscale.

        Returns:
            img (PIL image), grayscaled image.
        """
        return util.grayscale(img, num_output_channels=self.num_output_channels)


class RandomGrayscale:
    """
    Randomly convert the input image into grayscale image with a given probability.

    Args:
        prob (float, optional): Probability of the image being converted to grayscale (default=0.1).

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomGrayscale(0.3),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_prob
    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): PIL image to be converted to grayscale randomly.

        Returns:
            img (PIL image), Randomly apply grayscale to image, same number of channels as the input image.
                If input image has 1 channel, the output grayscale image is 1 channel.
                If input image has 3 channels, the output image has 3 identical grayscale channels.
        """
        if img.mode == 'L':
            num_output_channels = 1
        else:
            num_output_channels = 3

        if self.prob > random.random():
            return util.grayscale(img, num_output_channels=num_output_channels)
        return img


class Pad:
    """
    Pad the input PIL image according to padding parameters.

    Args:
        padding (Union[int, sequence]): The number of pixels to pad the image.
            If a single number is provided, pad all borders with this value.
            If a tuple or list of 2 values is provided, pad the left and top
            with the first value and the right and bottom with the second value.
            If 4 values are provided as a list or tuple,
            pad the left, top, right and bottom respectively.
        fill_value (Union[int, tuple], optional): The pixel intensity of the borders, only valid for
            padding_mode Border.CONSTANT (default=0).
            If it is an integer, it is used for all RGB channels.
            If it is a 3-tuple, it is used to fill R, G, B channels respectively.
        padding_mode (Border mode, optional): The method of padding (default=Border.CONSTANT).
            It can be any of [Border.CONSTANT, Border.EDGE, Border.REFLECT, Border.SYMMETRIC].

            - Border.CONSTANT, means it fills the border with constant values.

            - Border.EDGE, means it pads with the last value on the edge.

            - Border.REFLECT, means it reflects the values on the edge omitting the last
              value of edge.

            - Border.SYMMETRIC, means it reflects the values on the edge repeating the last
              value of edge.

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            # adds 10 pixels (default black) to each side of the border of the image
        ...                            py_vision.Pad(padding=10),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_pad
    def __init__(self, padding, fill_value=0, padding_mode=Border.CONSTANT):
        parse_padding(padding)

        self.padding = padding
        self.fill_value = fill_value
        self.padding_mode = DE_PY_BORDER_TYPE[padding_mode]
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to be padded.

        Returns:
            img (PIL image), Padded image.
        """
        return util.pad(img, self.padding, self.fill_value, self.padding_mode)


class RandomPerspective:
    """
    Randomly apply perspective transformation to the input PIL image with a given probability.

    Args:
        distortion_scale (float, optional): The scale of distortion, a float value between 0 and 1 (default=0.5).
        prob (float, optional): Probability of the image being applied perspective transformation (default=0.5).
        interpolation (Inter mode, optional): Image interpolation mode (default=Inter.BICUBIC).
            It can be any of [Inter.BILINEAR, Inter.NEAREST, Inter.BICUBIC].

            - Inter.BILINEAR, means the interpolation method is bilinear interpolation.

            - Inter.NEAREST, means the interpolation method is nearest-neighbor interpolation.

            - Inter.BICUBIC, means the interpolation method is bicubic interpolation.

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomPerspective(prob=0.1),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_random_perspective
    def __init__(self, distortion_scale=0.5, prob=0.5, interpolation=Inter.BICUBIC):
        self.distortion_scale = distortion_scale
        self.prob = prob
        self.interpolation = DE_PY_INTER_MODE[interpolation]

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): PIL image to apply perspective transformation randomly.

        Returns:
            img (PIL image), Image after being perspectively transformed randomly.
        """
        if not is_pil(img):
            raise ValueError("Input image should be a Pillow image.")
        if self.prob > random.random():
            start_points, end_points = util.get_perspective_params(img, self.distortion_scale)
            return util.perspective(img, start_points, end_points, self.interpolation)
        return img


class RandomErasing:
    """
    Erase the pixels, within a selected rectangle region, to the given value.

    Randomly applied on the input NumPy image array of shape (C, H, W) with a given probability.

    Zhun Zhong et al. 'Random Erasing Data Augmentation' 2017 See https://arxiv.org/pdf/1708.04896.pdf

    Args:
        prob (float, optional): Probability of applying RandomErasing (default=0.5).
        scale (sequence of floats, optional): Range of the relative erase area to the
            original image (default=(0.02, 0.33)).
        ratio (sequence of floats, optional): Range of the aspect ratio of the erase
            area (default=(0.3, 3.3)).
        value (Union[int, sequence, string]): Erasing value (default=0).
            If value is a single intieger, it is applied to all pixels to be erased.
            If value is a sequence of length 3, it is applied to R, G, B channels respectively.
            If value is a string 'random', the erase value will be obtained from a standard normal distribution.
        inplace (bool, optional): Apply this transform in-place (default=False).
        max_attempts (int, optional): The maximum number of attempts to propose a valid
            erase_area (default=10). If exceeded, return the original image.

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.ToTensor(),
        ...                            py_vision.RandomErasing(value='random')])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_random_erasing
    def __init__(self, prob=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False, max_attempts=10):
        self.prob = prob
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace
        self.max_attempts = max_attempts

    def __call__(self, np_img):
        """
        Call method.

        Args:
            np_img (numpy.ndarray): NumPy image array of shape (C, H, W) to be randomly erased.

        Returns:
            np_img (numpy.ndarray), Erased NumPy image array.
        """
        bounded = True
        if self.prob > random.random():
            i, j, erase_h, erase_w, erase_value = util.get_erase_params(np_img, self.scale, self.ratio,
                                                                        self.value, bounded, self.max_attempts)
            return util.erase(np_img, i, j, erase_h, erase_w, erase_value, self.inplace)
        return np_img


class Cutout:
    """
    Randomly cut (mask) out a given number of square patches from the input NumPy image array of shape (C, H, W).

    Terrance DeVries and Graham W. Taylor 'Improved Regularization of Convolutional Neural Networks with Cutout' 2017
    See https://arxiv.org/pdf/1708.04552.pdf

    Args:
        length (int): The side length of each square patch.
        num_patches (int, optional): Number of patches to be cut out of an image (default=1).

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.ToTensor(),
        ...                            py_vision.Cutout(80)])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_cutout
    def __init__(self, length, num_patches=1):
        self.length = length
        self.num_patches = num_patches
        self.random = False

    def __call__(self, np_img):
        """
        Call method.

        Args:
            np_img (numpy.ndarray): NumPy image array of shape (C, H, W) to be cut out.

        Returns:
            np_img (numpy.ndarray), NumPy image array with square patches cut out.
        """
        if not isinstance(np_img, np.ndarray):
            raise TypeError("img should be NumPy array. Got {}.".format(type(np_img)))
        if np_img.ndim != 3:
            raise TypeError('img dimension should be 3. Got {}.'.format(np_img.ndim))

        _, image_h, image_w = np_img.shape
        scale = (self.length * self.length) / (image_h * image_w)
        bounded = False

        for _ in range(self.num_patches):
            i, j, erase_h, erase_w, erase_value = util.get_erase_params(np_img, (scale, scale), (1, 1), 0, bounded,
                                                                        1)
            np_img = util.erase(np_img, i, j, erase_h, erase_w, erase_value)
        return np_img


class LinearTransformation:
    r"""
    Apply linear transformation to the input NumPy image array, given a square transformation matrix and
    a mean vector.

    The transformation first flattens the input array and subtracts the mean vector from it. It then computes
    the dot product with the transformation matrix, and reshapes it back to its original shape.

    Args:
        transformation_matrix (numpy.ndarray): a square transformation matrix of shape (D, D), where
            :math:`D = C \times H \times W`.
        mean_vector (numpy.ndarray): a NumPy ndarray of shape (D,) where :math:`D = C \times H \times W`.

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> import numpy as np
        >>> height, width = 32, 32
        >>> dim = 3 * height * width
        >>> transformation_matrix = np.ones([dim, dim])
        >>> mean_vector = np.zeros(dim)
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.Resize((height,width)),
        ...                            py_vision.ToTensor(),
        ...                            py_vision.LinearTransformation(transformation_matrix, mean_vector)])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_linear_transform
    def __init__(self, transformation_matrix, mean_vector):
        self.transformation_matrix = transformation_matrix
        self.mean_vector = mean_vector
        self.random = False

    def __call__(self, np_img):
        """
        Call method.

        Args:
            np_img (numpy.ndarray): NumPy image array of shape (C, H, W) to be linear transformed.

        Returns:
            np_img (numpy.ndarray), Linear transformed image.
        """
        return util.linear_transform(np_img, self.transformation_matrix, self.mean_vector)


class RandomAffine:
    """
    Apply Random affine transformation to the input PIL image.

    Args:
        degrees (Union[int, float, sequence]): Range of the rotation degrees.
            If degrees is a number, the range will be (-degrees, degrees).
            If degrees is a sequence, it should be (min, max).
        translate (sequence, optional): Sequence (tx, ty) of maximum translation in
            x(horizontal) and y(vertical) directions (default=None).
            The horizontal shift and vertical shift are selected randomly from the range:
            (-tx*width, tx*width) and (-ty*height, ty*height), respectively.
            If None, no translations are applied.
        scale (sequence, optional): Scaling factor interval (default=None, original scale is used).
        shear (Union[int, float, sequence], optional): Range of shear factor (default=None).
            If shear is an integer, then a shear parallel to the X axis in the range of (-shear, +shear) is applied.
            If shear is a tuple or list of size 2, then a shear parallel to the X axis in the range of
            (shear[0], shear[1]) is applied.
            If shear is a tuple of list of size 4, then a shear parallel to X axis in the range of
            (shear[0], shear[1]) and a shear parallel to Y axis in the range of (shear[2], shear[3]) is applied.
            If shear is None, no shear is applied.
        resample (Inter mode, optional): An optional resampling filter (default=Inter.NEAREST).
            If omitted, or if the image has mode "1" or "P", it is set to be Inter.NEAREST.
            It can be any of [Inter.BILINEAR, Inter.NEAREST, Inter.BICUBIC].

            - Inter.BILINEAR, means resample method is bilinear interpolation.

            - Inter.NEAREST, means resample method is nearest-neighbor interpolation.

            - Inter.BICUBIC, means resample method is bicubic interpolation.

        fill_value (Union[tuple, int], optional): Optional filling value to fill the area outside the transform
            in the output image. There must be three elements in the tuple and the value of a single element is
            within the range [0, 255].
            Used only in Pillow versions > 5.0.0 (default=0, filling is performed).

    Raises:
        ValueError: If degrees is negative.
        ValueError: If translation value is not between 0 and 1.
        ValueError: If scale is not positive.
        ValueError: If shear is a number but is not positive.
        TypeError: If degrees is not a number or a list or a tuple.
            If degrees is a list or tuple, its length is not 2.
        TypeError: If translate is specified but is not list or a tuple of length 2.
        TypeError: If scale is not a list or tuple of length 2.
        TypeError: If shear is not a list or tuple of length 2 or 4.
        TypeError: If fill_value is not a single integer or a 3-tuple.

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_random_affine
    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=Inter.NEAREST, fill_value=0):
        # Parameter checking
        # rotation
        if shear is not None:
            if isinstance(shear, numbers.Number):
                shear = (-1 * shear, shear)
            else:
                if len(shear) == 2:
                    shear = [shear[0], shear[1], 0., 0.]
                elif len(shear) == 4:
                    shear = [s for s in shear]

        if isinstance(degrees, numbers.Number):
            degrees = (-degrees, degrees)

        self.degrees = degrees
        self.translate = translate
        self.scale_ranges = scale
        self.shear = shear
        self.resample = DE_PY_INTER_MODE[resample]
        self.fill_value = fill_value

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to apply affine transformation.

        Returns:
            img (PIL image), Randomly affine transformed image.
        """

        return util.random_affine(img,
                                  self.degrees,
                                  self.translate,
                                  self.scale_ranges,
                                  self.shear,
                                  self.resample,
                                  self.fill_value)


class MixUp:
    """
    Apply mix up transformation to the input image and label. Make one input data combined with others.

    Args:
        batch_size (int): Batch size of dataset.
        alpha (float):  Mix up rate.
        is_single (bool): Identify if single batch or multi-batch mix up transformation is to be used
            (Default=True, which is single batch).


    Examples:
        >>> # Setup multi-batch mixup transformation
        >>> transform = [py_vision.MixUp(batch_size=16, alpha=0.2, is_single=False)]
        >>> # Apply the transform to the dataset through dataset.map()
        >>> image_folder_dataset = image_folder_dataset.map(input_columns="image",
        ...                                                 operations=transform)
    """

    @check_mix_up
    def __init__(self, batch_size, alpha, is_single=True):
        self.image = 0
        self.label = 0
        self.is_first = True
        self.batch_size = batch_size
        self.alpha = alpha
        self.is_single = is_single
        self.random = False

    def __call__(self, image, label):
        """
        Call method.

        Args:
            image (numpy.ndarray): NumPy image to apply mix up transformation.
            label(numpy.ndarray): NumPy label to apply mix up transformation.

        Returns:
            image (numpy.ndarray): NumPy image after applying mix up transformation.
            label(numpy.ndarray): NumPy label after applying mix up transformation.
        """
        if self.is_single:
            return util.mix_up_single(self.batch_size, image, label, self.alpha)
        return util.mix_up_muti(self, self.batch_size, image, label, self.alpha)


class RgbToHsv:
    """
    Convert a NumPy RGB image or a batch of NumPy RGB images to HSV images.

    Args:
        is_hwc (bool): The flag of image shape, (H, W, C) or (N, H, W, C) if True
                       and (C, H, W) or (N, C, H, W) if False (default=False).

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.CenterCrop(20),
        ...                            py_vision.ToTensor(),
        ...                            py_vision.RgbToHsv()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    def __init__(self, is_hwc=False):
        self.is_hwc = is_hwc
        self.random = False

    def __call__(self, rgb_imgs):
        """
        Call method.

        Args:
            rgb_imgs (numpy.ndarray): NumPy RGB images array of shape (H, W, C) or (N, H, W, C),
                                      or (C, H, W) or (N, C, H, W) to be converted.

        Returns:
            np_hsv_img (numpy.ndarray), NumPy HSV images with same shape of rgb_imgs.
        """
        return util.rgb_to_hsvs(rgb_imgs, self.is_hwc)


class HsvToRgb:
    """
    Convert a NumPy HSV image or one batch NumPy HSV images to RGB images.

    Args:
        is_hwc (bool): The flag of image shape, (H, W, C) or (N, H, W, C) if True
                       and (C, H, W) or (N, C, H, W) if False (default=False).

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.CenterCrop(20),
        ...                            py_vision.ToTensor(),
        ...                            py_vision.HsvToRgb()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    def __init__(self, is_hwc=False):
        self.is_hwc = is_hwc
        self.random = False

    def __call__(self, hsv_imgs):
        """
        Call method.

        Args:
            hsv_imgs (numpy.ndarray): NumPy HSV images array of shape (H, W, C) or (N, H, W, C),
                                      or (C, H, W) or (N, C, H, W) to be converted.

        Returns:
            rgb_imgs (numpy.ndarray), NumPy RGB image with same shape of hsv_imgs.
        """
        return util.hsv_to_rgbs(hsv_imgs, self.is_hwc)


class RandomColor:
    """
    Adjust the color of the input PIL image by a random degree.

    Args:
        degrees (sequence): Range of random color adjustment degrees.
            It should be in (min, max) format (default=(0.1,1.9)).

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomColor((0.5, 2.0)),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_positive_degrees
    def __init__(self, degrees=(0.1, 1.9)):
        self.degrees = degrees

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to be color adjusted.

        Returns:
            img (PIL image), Color adjusted image.
        """

        return util.random_color(img, self.degrees)


class RandomSharpness:
    """
    Adjust the sharpness of the input PIL image by a fixed or random degree. Degree of 0.0 gives a blurred image,
    degree of 1.0 gives the original image, and degree of 2.0 gives a sharpened image.

    Args:
        degrees (sequence): Range of random sharpness adjustment degrees.
            It should be in (min, max) format (default=(0.1,1.9)).

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomSharpness((0.5, 1.5)),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_positive_degrees
    def __init__(self, degrees=(0.1, 1.9)):
        self.degrees = degrees

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to be sharpness adjusted.

        Returns:
            img (PIL image), Color adjusted image.
        """

        return util.random_sharpness(img, self.degrees)


class AutoContrast:
    """
    Automatically maximize the contrast of the input PIL image.

    Args:
        cutoff (float, optional): Percent of pixels to cut off from the histogram,
            the value must be in the range [0.0, 50.0) (default=0.0).
        ignore (Union[int, sequence], optional): Pixel values to ignore (default=None).

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.AutoContrast(),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_auto_contrast
    def __init__(self, cutoff=0.0, ignore=None):
        self.cutoff = cutoff
        self.ignore = ignore
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to be augmented with AutoContrast.

        Returns:
            img (PIL image), Augmented image.
        """

        return util.auto_contrast(img, self.cutoff, self.ignore)


class Invert:
    """
    Invert colors of input PIL image.

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.Invert(),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    def __init__(self):
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to be color Inverted.

        Returns:
            img (PIL image), Color inverted image.
        """

        return util.invert_color(img)


class Equalize:
    """
    Equalize the histogram of input PIL image.

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.Equalize(),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")

    """

    def __init__(self):
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to be equalized.

        Returns:
            img (PIL image), Equalized image.
        """

        return util.equalize(img)


class UniformAugment:
    """
    Uniformly select and apply a number of transforms sequentially from
    a list of transforms. Randomly assign a probability to each transform for
    each image to decide whether to apply the transform or not.

    All the transforms in transform list must have the same input/output data type.

    Args:
         transforms (list): List of transformations to be chosen from to apply.
         num_ops (int, optional): number of transforms to sequentially apply (default=2).

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms = [py_vision.CenterCrop(64),
        ...               py_vision.RandomColor(),
        ...               py_vision.RandomSharpness(),
        ...               py_vision.RandomRotation(30)]
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.UniformAugment(transforms),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_uniform_augment_py
    def __init__(self, transforms, num_ops=2):
        self.transforms = transforms
        self.num_ops = num_ops
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to apply transformation.

        Returns:
            img (PIL image), Transformed image.
        """
        return util.uniform_augment(img, self.transforms.copy(), self.num_ops)
