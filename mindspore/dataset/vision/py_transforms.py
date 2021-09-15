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
The module vision.py_transforms is mainly implemented based on Python PIL, which
provides many kinds of image augmentation methods and conversion methods between
PIL image and numpy.ndarray. For users who prefer using Python PIL in computer vision
tasks, this module is a good choice to process images. Users can also self-define
their own augmentation methods with Python PIL.
"""
import numbers
import random

import numpy as np
from PIL import Image

import mindspore.dataset.transforms.py_transforms as py_transforms
from . import py_transforms_util as util
from .c_transforms import parse_padding
from .validators import check_prob, check_center_crop, check_five_crop, check_resize_interpolation, check_random_resize_crop, \
    check_normalize_py, check_normalizepad_py, check_random_crop, check_random_color_adjust, check_random_rotation, \
    check_ten_crop, check_num_channels, check_pad, check_rgb_to_hsv, check_hsv_to_rgb, \
    check_random_perspective, check_random_erasing, check_cutout, check_linear_transform, check_random_affine, \
    check_mix_up, check_positive_degrees, check_uniform_augment_py, check_auto_contrast, check_rgb_to_bgr, \
    check_adjust_gamma
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
    """
    Specify the function as "not random", i.e., it produces deterministic result.
    A Python function can only be cached after it is specified as "not random".
    """
    function.random = False
    return function


class ToTensor(py_transforms.PyTensorOperation):
    """
    Convert the input PIL Image or numpy.ndarray of shape (H, W, C) in the range [0, 255] to numpy.ndarray of
    shape (C, H, W) in the range [0.0, 1.0] with the desired dtype.

    Note:
        The values in the input image will be rescaled from [0, 255] to [0.0, 1.0].
        The dtype will be cast to `output_type`.
        The number of channels remains the same.

    Args:
        output_type (numpy.dtype, optional): The dtype of the numpy.ndarray output (default=np.float32).

    Raises:
        TypeError: If the input is not PIL Image or numpy.ndarray.
        TypeError: If the dimension of input is not 2 or 3.

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> # create a list of transformations to be applied to the "image" column of each data row
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomHorizontalFlip(0.5),
        ...                            py_vision.ToTensor()])
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
            img (Union[PIL Image, numpy.ndarray]): PIL Image or numpy.ndarray to be type converted.

        Returns:
            numpy.ndarray, converted numpy.ndarray with desired type.
        """
        return util.to_tensor(img, self.output_type)


class ToType(py_transforms.PyTensorOperation):
    """
    Convert the input numpy.ndarray image to the desired dtype.

    Args:
        output_type (numpy.dtype): The dtype of the numpy.ndarray output, e.g. numpy.float32.

    Raises:
        TypeError: If the input is not numpy.ndarray.

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
            img (numpy.ndarray): numpy.ndarray to be dtype converted.

        Returns:
            numpy.ndarray, converted numpy.ndarray with desired dtype.
        """
        return util.to_type(img, self.output_type)


class HWC2CHW(py_transforms.PyTensorOperation):
    """
    Transpose the input numpy.ndarray image of shape (H, W, C) to (C, H, W).

    Raises:
        TypeError: If the input is not numpy.ndarray.
        TypeError: If the dimension of input is not 3.

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
            img (numpy.ndarray): numpy.ndarray of shape (H, W, C) to be transposed.

        Returns:
            numpy.ndarray, transposed numpy.ndarray of shape (C, H, W).
        """
        return util.hwc_to_chw(img)


class ToPIL(py_transforms.PyTensorOperation):
    """
    Convert the input decoded numpy.ndarray image to PIL Image.

    Note:
        The conversion mode will be determined from type according to `PIL.Image.fromarray`.

    Raises:
        TypeError: If the input is not numpy.ndarray or PIL Image.

    Examples:
        >>> # data is already decoded, but not in PIL Image format
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
            img (numpy.ndarray): Decoded numpy.ndarray image to be converted to PIL Image.

        Returns:
            PIL Image, converted PIL Image.
        """
        return util.to_pil(img)


class Decode(py_transforms.PyTensorOperation):
    """
    Decode the input raw image to PIL Image format in RGB mode.

    Raises:
        ValueError: If the input is not raw data.
        ValueError: If the input image is already decoded.

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
            img (Bytes-like Object): Raw image data to be decoded.

        Returns:
            PIL Image, decoded PIL Image in RGB mode.
        """
        return util.decode(img)


class Normalize(py_transforms.PyTensorOperation):
    r"""
    Normalize the input numpy.ndarray image of shape (C, H, W) with the specified mean and standard deviation.

    .. math::

        output_{c} = \frac{input_{c} - mean_{c}}{std_{c}}

    Note:
        The values of the input image need to be in the range [0.0, 1.0]. If not so, call `ToTensor` first.

    Args:
        mean (Union[float, sequence]): list or tuple of mean values for each channel, arranged in channel order. The
            values must be in the range [0.0, 1.0].
            If a single float is provided, it will be filled to the same length as the channel.
        std (Union[float, sequence]): list or tuple of standard deviation values for each channel, arranged in channel
            order. The values must be in the range (0.0, 1.0].
            If a single float is provided, it will be filled to the same length as the channel.

    Raises:
        TypeError: If the input is not numpy.ndarray.
        TypeError: If the dimension of input is not 3.
        NotImplementedError: If the dtype of input is a subdtype of np.integer.
        ValueError: If the length of the mean and std are not equal.
        ValueError: If the length of the mean or std is neither equal to the channel length nor 1.

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
            img (numpy.ndarray): numpy.ndarray to be normalized.

        Returns:
            numpy.ndarray, normalized numpy.ndarray.
        """
        return util.normalize(img, self.mean, self.std)


class NormalizePad(py_transforms.PyTensorOperation):
    r"""
    Normalize the input numpy.ndarray image of shape (C, H, W) with the specified mean and standard deviation,
    then pad an extra channel filled with zeros.

    .. math::
        output_{c} = \begin{cases}
        \frac{input_{c} - mean_{c}}{std_{c}}, & \text{if} \quad 0 \le c < 3 \text{;}\\
        0, & \text{if} \quad c = 3 \text{.}
        \end{cases}

    Note:
        The values of the input image need to be in the range [0.0, 1.0]. If not so, call `ToTensor` first.

    Args:
        mean (Union[float, sequence]): list or tuple of mean values for each channel, arranged in channel order. The
            values must be in the range [0.0, 1.0].
            If a single float is provided, it will be filled to the same length as the channel.
        std (Union[float, sequence]): list or tuple of standard deviation values for each channel, arranged in channel
            order. The values must be in the range (0.0, 1.0].
            If a single float is provided, it will be filled to the same length as the channel.
        dtype (str): The dtype of the numpy.ndarray output when `pad_channel` is set True. Only "float32" and "float16"
            are supported (default="float32").

    Raises:
        TypeError: If the input is not numpy.ndarray.
        TypeError: If the dimension of input is not 3.
        NotImplementedError: If the dtype of input is a subdtype of np.integer.
        ValueError: If the length of the mean and std are not equal.
        ValueError: If the length of the mean or std is neither equal to the channel length nor 1.

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
            img (numpy.ndarray): numpy.ndarray to be normalized and padded.

        Returns:
            numpy.ndarray, normalized and padded numpy.ndarray.
        """
        return util.normalize(img, self.mean, self.std, pad_channel=True, dtype=self.dtype)


class RandomCrop(py_transforms.PyTensorOperation):
    """
    Crop the input PIL Image at a random location with the specified size.

    Args:
        size (Union[int, sequence]): The output size of the cropped image.
            If size is an integer, a square of size (size, size) is returned.
            If size is a sequence of length 2, it should be in shape of (height, width).
        padding (Union[int, sequence], optional): Padding on each border of the image (default=None).
            If padding is not None, pad the image before cropping.
            If a single number is provided, pad all borders with this value.
            If a sequence of length 2 is provided, pad the left/top border
            with the first value and right/bottom border with the second value.
            If a sequence of length 4 is provided, pad the left, top, right and bottom borders respectively.
        pad_if_needed (bool, optional): Pad the image if either side is smaller than
            the given output size (default=False).
        fill_value (Union[int, tuple], optional): Pixel fill value to pad the borders when padding_mode is
            Border.CONSTANT (default=0). If a tuple of length 3 is provided, it is used to fill R, G, B
            channels respectively.
        padding_mode (Border, optional): The method of padding (default=Border.CONSTANT). It can be any of
            [Border.CONSTANT, Border.EDGE, Border.REFLECT, Border.SYMMETRIC].

            - Border.CONSTANT, means to pad with given constant values.

            - Border.EDGE, means to pad with the last value at the edge.

            - Border.REFLECT, means to pad with reflection of image omitting the last value at the edge.

            - Border.SYMMETRIC, means to pad with reflection of image repeating the last value at the edge.

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
            img (PIL Image): Image to be randomly cropped.

        Returns:
            PIL Image, cropped image.
        """
        return util.random_crop(img, self.size, self.padding, self.pad_if_needed,
                                self.fill_value, self.padding_mode)


class RandomHorizontalFlip(py_transforms.PyTensorOperation):
    """
    Randomly flip the input image horizontally with a given probability.

    Args:
        prob (float, optional): Probability of the image to be horizontally flipped (default=0.5).

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
            img (PIL Image): Image to be horizontally flipped.

        Returns:
            PIL Image, randomly horizontally flipped image.
        """
        return util.random_horizontal_flip(img, self.prob)


class RandomVerticalFlip(py_transforms.PyTensorOperation):
    """
    Randomly flip the input image vertically with a given probability.

    Args:
        prob (float, optional): Probability of the image to be vertically flipped (default=0.5).

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
            img (PIL Image): Image to be vertically flipped.

        Returns:
            PIL Image, randomly vertically flipped image.
        """
        return util.random_vertical_flip(img, self.prob)


class Resize(py_transforms.PyTensorOperation):
    """
    Resize the input PIL Image to the given size.

    Args:
        size (Union[int, sequence]): The output size of the image.
            If size is an integer, the smaller edge of the image will be resized to this
            value, keeping the image aspect ratio the same.
            If size is a sequence of length 2, it should be in shape of (height, width).
        interpolation (Inter, optional): Image interpolation mode (default=Inter.BILINEAR).
            It can be any of [Inter.NEAREST, Inter.ANTIALIAS, Inter.BILINEAR, Inter.BICUBIC].

            - Inter.NEAREST, nearest-neighbor interpolation.

            - Inter.ANTIALIAS, antialias interpolation.

            - Inter.BILINEAR, bilinear interpolation.

            - Inter.BICUBIC, bicubic interpolation.

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
            img (PIL Image): Image to be resized.

        Returns:
            PIL Image, resized image.
        """
        return util.resize(img, self.size, self.interpolation)


class RandomResizedCrop(py_transforms.PyTensorOperation):
    """
    Randomly crop the image and resize it to a given size.

    Args:
        size (Union[int, sequence]): The size of the output image.
            If size is an integer, a square of size (size, size) is returned.
            If size is a sequence of length 2, it should be in shape of (height, width).
        scale (Union[list, tuple], optional): Respective size range of the original image to be cropped
            in shape of (min, max) (default=(0.08, 1.0)).
        ratio (Union[list, tuple], optional): Aspect ratio range to be cropped
            in shape of (min, max) (default=(3./4., 4./3.)).
        interpolation (Inter, optional): Image interpolation mode (default=Inter.BILINEAR).
            It can be any of [Inter.NEAREST, Inter.ANTIALIAS, Inter.BILINEAR, Inter.BICUBIC].

            - Inter.NEAREST, nearest-neighbor interpolation.

            - Inter.ANTIALIAS, antialias interpolation.

            - Inter.BILINEAR, bilinear interpolation.

            - Inter.BICUBIC, bicubic interpolation.

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
            img (PIL Image): Image to be randomly cropped and resized.

        Returns:
            PIL Image, randomly cropped and resized image.
        """
        return util.random_resize_crop(img, self.size, self.scale, self.ratio,
                                       self.interpolation, self.max_attempts)


class CenterCrop(py_transforms.PyTensorOperation):
    """
    Crop the central reigion of the input PIL Image with the given size.

    Args:
        size (Union[int, sequence]): The output size of the cropped image.
            If size is an integer, a square of size (size, size) is returned.
            If size is a sequence of length 2, it should be in shape of (height, width).

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.CenterCrop(64),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_center_crop
    def __init__(self, size):
        self.size = size
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL Image): Image to be center cropped.

        Returns:
            PIL Image, cropped image.
        """
        return util.center_crop(img, self.size)


class RandomColorAdjust(py_transforms.PyTensorOperation):
    """
    Randomly adjust the brightness, contrast, saturation, and hue of the input PIL Image.

    Args:
        brightness (Union[float, tuple], optional): Brightness adjustment factor,
            which must be non negative (default=(1, 1)).
            If brightness is a float, the factor is uniformly chosen in range of [max(0, 1-brightness), 1+brightness].
            If brightness is a sequence of length 2, it should be in shape of [min, max].
        contrast (Union[float, tuple], optional): Contrast adjustment factor,
            which must be non negative (default=(1, 1)).
            If contrast is a float, the factor is uniformly chosen in range of [max(0, 1-contrast), 1+contrast].
            If contrast is a sequence of length 2, it should be in shape of [min, max].
        saturation (Union[float, tuple], optional): Saturation adjustment factor,
            which must be non negative(default=(1, 1)).
            If saturation is a float, the factor is uniformly chosen in range of [max(0, 1-saturation), 1+saturation].
            If saturation is a sequence of length 2, it should be in shape of [min, max].
        hue (Union[float, tuple], optional): Hue adjustment factor (default=(0, 0)).
            If hue is a float, the range will be [-hue, hue], where 0 <= hue <= 0.5.
            If hue is a sequence of length 2, it should be in shape of [min, max], where -0.5 <= min <= max <= 0.5.

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
            img (PIL image): Image to be randomly color adjusted.

        Returns:
            PIL Image, randomly color adjusted image.
        """
        return util.random_color_adjust(img, self.brightness, self.contrast, self.saturation, self.hue)


class RandomRotation(py_transforms.PyTensorOperation):
    """
    Rotate the input PIL Image by a random angle.

    Note:
        See https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.rotate.

    Args:
        degrees (Union[int, float, sequence]): Range of random rotation degrees.
            If degrees is a number, the range will be converted to (-degrees, degrees).
            If degrees is a sequence of length 2, it should be in shape of (min, max).
        resample (Inter, optional): An optional resampling filter (default=Inter.NEAREST).
            If the image is in mode of "1" or "P", it is set to Inter.NEAREST by default.
            It can be any of [Inter.NEAREST, Inter.ANTIALIAS, Inter.BILINEAR, Inter.BICUBIC].

            - Inter.NEAREST, nearest-neighbor interpolation.

            - Inter.ANTIALIAS, antialias interpolation.

            - Inter.BILINEAR, bilinear interpolation.

            - Inter.BICUBIC, bicubic interpolation.

        expand (bool, optional): Optional expansion flag (default=False).
            If set to True, expand the output image to make it large enough to hold the entire rotated image.
            If set to False, keep the output image the same size as the input.
            Note that the expand flag assumes rotation around the center and no translation.
        center (tuple, optional): Optional center of rotation, which must be a tuple of length 2 (default=None).
            Origin is the top left corner. Default None means to set the center of the image.
        fill_value (int or tuple, optional): Pixel fill value for the area outside the rotated image (default=0).
            If fill_value is a tuple of length 3, it is used to fill R, G, B channels respectively.
            If fill_value is an integer, it is used to fill all RGB channels.

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
            img (PIL Image): Image to be randomly rotated.

        Returns:
            PIL Image, randomly rotated image.
        """
        return util.random_rotation(img, self.degrees, self.resample, self.expand, self.center, self.fill_value)


class FiveCrop(py_transforms.PyTensorOperation):
    """
    Crop the given image into one central crop and four corners.

    Args:
        size (Union[int, sequence]): The output size of the cropped images.
            If size is an integer, a square of size (size, size) is returned.
            If size is a sequence of length 2, it should be in shape of (height, width).

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

    @check_five_crop
    def __init__(self, size):
        self.size = size
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            tuple, a tuple of five PIL Image in order of top_left, top_right, bottom_left, bottom_right, center.
        """
        return util.five_crop(img, self.size)


class TenCrop(py_transforms.PyTensorOperation):
    """
    Crop the given image into one central crop and four corners plus the flipped version of these.

    Args:
        size (Union[int, sequence]): The output size of the cropped images.
            If size is an integer, a square of size (size, size) is returned.
            If size is a sequence of length 2, it should be in shape of (height, width).
        use_vertical_flip (bool, optional): Whether to flip the image vertically,
            otherwise horizontally (default=False).

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
            img (PIL Image): Image to be cropped.

        Returns:
            tuple, a tuple of 10 PIL Image, in order of top_left, top_right, bottom_left, bottom_right, center
                of the original image and top_left, top_right, bottom_left, bottom_right, center of the flipped image.
        """
        return util.ten_crop(img, self.size, self.use_vertical_flip)


class Grayscale(py_transforms.PyTensorOperation):
    """
    Convert the input PIL Image to grayscale.

    Args:
        num_output_channels (int): Number of channels of the output grayscale image, which can be 1 or 3 (default=1).
            If num_output_channels is 3, the returned image will have 3 identical RGB channels.

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
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image, converted grayscale image.
        """
        return util.grayscale(img, num_output_channels=self.num_output_channels)


class RandomGrayscale(py_transforms.PyTensorOperation):
    """
    Randomly convert the input image into grayscale with a given probability.

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
            img (PIL Image): Image to be randomly converted to grayscale.

        Returns:
            PIL Image, randomly converted grayscale image, which has the same number of channels as the input image.
                If input image has 1 channel, the output grayscale image will have 1 channel.
                If input image has 3 channels, the output grayscale image will have 3 identical channels.
        """
        if img.mode == 'L':
            num_output_channels = 1
        else:
            num_output_channels = 3

        if self.prob > random.random():
            return util.grayscale(img, num_output_channels=num_output_channels)
        return img


class Pad(py_transforms.PyTensorOperation):
    """
    Pad the input image on all sides with the given padding parameters.

    Args:
        padding (Union[int, sequence]): The number of pixels padded on the image borders.
            If a single number is provided, pad all borders with this value.
            If a sequence of length 2 is provided, pad the left and top with the
            first value and the right and bottom with the second value.
            If a sequence of length 4 is provided, pad the left, top, right and bottom respectively.
        fill_value (Union[int, tuple], optional): Pixel fill value to pad the borders,
            only valid when padding_mode is Border.CONSTANT (default=0).
            If fill_value is an integer, it is used for all RGB channels.
            If fill_value is a tuple of length 3, it is used to fill R, G, B channels respectively.
        padding_mode (Border, optional): The method of padding (default=Border.CONSTANT).
            It can be any of [Border.CONSTANT, Border.EDGE, Border.REFLECT, Border.SYMMETRIC].

            - Border.CONSTANT, pads with a constant value.

            - Border.EDGE, pads with the last value at the edge of the image.

            - Border.REFLECT, pads with reflection of the image omitting the last value on the edge.

            - Border.SYMMETRIC, pads with reflection of the image repeating the last value on the edge.

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            # adds 10 pixels (default black) to each border of the image
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
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image, padded image.
        """
        return util.pad(img, self.padding, self.fill_value, self.padding_mode)


class RandomPerspective(py_transforms.PyTensorOperation):
    """
    Randomly apply perspective transformation to the input PIL Image with a given probability.

    Args:
        distortion_scale (float, optional): The scale of distortion, in range of [0, 1] (default=0.5).
        prob (float, optional): Probability of the image being applied perspective transformation (default=0.5).
        interpolation (Inter, optional): Image interpolation mode (default=Inter.BICUBIC).
            It can be any of [Inter.BILINEAR, Inter.NEAREST, Inter.BICUBIC].

            - Inter.BILINEAR, bilinear interpolation.

            - Inter.NEAREST, nearest-neighbor interpolation.

            - Inter.BICUBIC, bicubic interpolation.

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
            img (PIL Image): Image to be applied randomly perspective transformation.

        Returns:
            PIL Image, image applied randomly perspective transformation.
        """
        if not is_pil(img):
            raise ValueError("Input image should be a Pillow image.")
        if self.prob > random.random():
            start_points, end_points = util.get_perspective_params(
                img, self.distortion_scale)
            return util.perspective(img, start_points, end_points, self.interpolation)
        return img


class RandomErasing(py_transforms.PyTensorOperation):
    """
    Randomly erase the pixels within a random selected rectangle region with a given probability.

    See Zhun Zhong et al. 'Random Erasing Data Augmentation' 2017 on https://arxiv.org/pdf/1708.04896.pdf

    Args:
        prob (float, optional): Probability of the image being randomly erased (default=0.5).
        scale (sequence of floats, optional): Range of the relative erase area to the
            original image (default=(0.02, 0.33)).
        ratio (sequence, optional): Range of aspect ratio of the erased area (default=(0.3, 3.3)).
        value (Union[int, sequence, str]): Erasing value (default=0).
            If value is a single integer, it is used to erase all pixels.
            If value is a sequence of length 3, it is used to erase R, G, B channels respectively.
            If value is a string of 'random', each pixel will be erased with a random value obtained
            from a standard normal distribution.
        inplace (bool, optional): Whether to apply this transformation inplace (default=False).
        max_attempts (int, optional): The maximum number of attempts to propose a valid
            area to be erased (default=10). If exceeded, return the original image.

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
            np_img (numpy.ndarray): image in shape of (C, H, W) to be randomly erased.

        Returns:
            numpy.ndarray, erased image.
        """
        bounded = True
        if self.prob > random.random():
            i, j, erase_h, erase_w, erase_value = util.get_erase_params(np_img, self.scale, self.ratio,
                                                                        self.value, bounded, self.max_attempts)
            return util.erase(np_img, i, j, erase_h, erase_w, erase_value, self.inplace)
        return np_img


class Cutout(py_transforms.PyTensorOperation):
    """
    Randomly apply a given number of square patches of zeros to a location within the input
    numpy.ndarray image of shape (C, H, W).

    See Terrance DeVries and Graham W. Taylor 'Improved Regularization of Convolutional Neural Networks with Cutout'
    2017 on https://arxiv.org/pdf/1708.04552.pdf

    Args:
        length (int): The side length of each square patch.
        num_patches (int, optional): Number of patches to be applied to the image (default=1).

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
            np_img (numpy.ndarray): Image in shape of (C, H, W) to be cut out.

        Returns:
            numpy.ndarray, image cut out.
        """
        if not isinstance(np_img, np.ndarray):
            raise TypeError(
                "img should be NumPy array. Got {}.".format(type(np_img)))
        if np_img.ndim != 3:
            raise TypeError(
                'img dimension should be 3. Got {}.'.format(np_img.ndim))

        _, image_h, image_w = np_img.shape
        scale = (self.length * self.length) / (image_h * image_w)
        bounded = False

        for _ in range(self.num_patches):
            i, j, erase_h, erase_w, erase_value = util.get_erase_params(np_img, (scale, scale), (1, 1), 0, bounded,
                                                                        1)
            np_img = util.erase(np_img, i, j, erase_h, erase_w, erase_value)
        return np_img


class LinearTransformation(py_transforms.PyTensorOperation):
    r"""
    Transform the input numpy.ndarray image with a given square transformation matrix and a mean vector.
    It will first flatten the input image and subtract the mean vector from it, then compute the dot
    product with the transformation matrix, finally reshape it back to its original shape.

    Args:
        transformation_matrix (numpy.ndarray): A square transformation matrix in shape of (D, D), where
            :math:`D = C \times H \times W`.
        mean_vector (numpy.ndarray): A mean vector in shape of (D,), where :math:`D = C \times H \times W`.

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
            np_img (numpy.ndarray): Image in shape of (C, H, W) to be linearly transformed.

        Returns:
            numpy.ndarray, linearly transformed image.
        """
        return util.linear_transform(np_img, self.transformation_matrix, self.mean_vector)


class RandomAffine(py_transforms.PyTensorOperation):
    """
    Apply random affine transformation to the input PIL Image.

    Args:
        degrees (Union[int, float, sequence]): Range of degrees to select from.
            If degrees is a number, the range will be (-degrees, degrees).
            If degrees is a sequence, it should be in shape of (min, max).
        translate (sequence, optional): Maximum absolute fraction sequence in shape of (tx, ty)
            for horizontal and vertical translations. The horizontal and vertical shifts are randomly
            selected in the range (-tx * width, tx * width) and (-ty * height, ty * height) respectively.
            (default=None, no translation will be applied).
        scale (sequence, optional): Scaling factor interval (default=None, keep original scale).
        shear (Union[int, float, sequence], optional): Range of shear factor to select from.
            If shear is an integer, a shear parallel to the X axis in the range (-shear, shear) will be applied.
            If shear is a sequence of length 2, a shear parallel to the X axis in the range (shear[0], shear[1])
            will be applied.
            If shear is a sequence of length 4, a shear parallel to the X axis in the range (shear[0], shear[1])
            and a shear parallel to the Y axis in the range (shear[2], shear[3]) will be applied.
            (default=None, no shear will be applied).
        resample (Inter, optional): An optional resampling filter (default=Inter.NEAREST).
            If the PIL Image is in mode of "1" or "P", it is set to Inter.NEAREST by default.
            It can be any of [Inter.BILINEAR, Inter.NEAREST, Inter.BICUBIC].

            - Inter.BILINEAR, bilinear interpolation.

            - Inter.NEAREST, nearest-neighbor interpolation.

            - Inter.BICUBIC, bicubic interpolation.

        fill_value (Union[int, tuple], optional): Pixel fill value for the area outside the
            transformed image (default=0).
            If fill_value is an integer, it is used for all RGB channels.
            If fill_value is a tuple of length 3, it is used to fill R, G, B channels respectively.
            Only supported with Pillow version > 5.0.0.

    Raises:
        ValueError: If degrees is negative.
        ValueError: If translation is not between 0 and 1.
        ValueError: If scale is not positive.
        ValueError: If shear is a non positive number.
        TypeError: If degrees is not a number or a sequence of length 2.
        TypeError: If translate is defined but not a sequence of length 2.
        TypeError: If scale is not a sequence of length 2.
        TypeError: If shear is not a sequence of length 2 or 4.
        TypeError: If fill_value is not an integer or a tuple of length 3.

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
            img (PIL Image): Image to be randomly affine transformed.

        Returns:
            PIL Image, randomly affine transformed image.
        """

        return util.random_affine(img,
                                  self.degrees,
                                  self.translate,
                                  self.scale_ranges,
                                  self.shear,
                                  self.resample,
                                  self.fill_value)


class MixUp(py_transforms.PyTensorOperation):
    """
    Randomly mix up a batch of images together with its labels. Each image will be multiplied by a random
    weight lambda generated from the beta distribution and then added to another image multiplied by 1 - lambda.
    The same transformation will be applied to their labels with the same value of lambda. Make sure that the
    labels are one hot encoded in advance.

    Args:
        batch_size (int): The number of images in a batch.
        alpha (float): The alpha and beta parameter in the beta distribution.
        is_single (bool, optional): If True, it will randomly mix up [img(0), ..., img(n-1), img(n)] with
            [img1, ..., img(n), img0] in each batch. Otherwise, it will randomly mix up images with the
            output of mixing of previous batch of images (Default=True).


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
            image (numpy.ndarray): Images to be mixed up.
            label (numpy.ndarray): Labels to be mixed up.

        Returns:
            numpy.ndarray, images after mixing up.
            numpy.ndarray, labels after mixing up.
        """
        if self.is_single:
            return util.mix_up_single(self.batch_size, image, label, self.alpha)
        return util.mix_up_muti(self, self.batch_size, image, label, self.alpha)


class RgbToBgr(py_transforms.PyTensorOperation):
    """
    Convert one or more numpy.ndarray images from RGB to BGR.

    Args:
        is_hwc (bool): Whether the image is in shape of (H, W, C) or (N, H, W, C), otherwise
            in shape of (C, H, W) or (N, C, H, W) (default=False).

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.CenterCrop(20),
        ...                            py_vision.ToTensor(),
        ...                            py_vision.RgbToBgr()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_rgb_to_bgr
    def __init__(self, is_hwc=False):
        self.is_hwc = is_hwc
        self.random = False

    def __call__(self, rgb_imgs):
        """
        Call method.

        Args:
            rgb_imgs (numpy.ndarray): RGB images to be converted.

        Returns:
            numpy.ndarray, converted BGR images.
        """
        return util.rgb_to_bgrs(rgb_imgs, self.is_hwc)


class RgbToHsv(py_transforms.PyTensorOperation):
    """
    Convert one or more numpy.ndarray images from RGB to HSV.

    Args:
        is_hwc (bool): Whether the image is in shape of (H, W, C) or (N, H, W, C), otherwise
            in shape of (C, H, W) or (N, C, H, W) (default=False).

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

    @check_rgb_to_hsv
    def __init__(self, is_hwc=False):
        self.is_hwc = is_hwc
        self.random = False

    def __call__(self, rgb_imgs):
        """
        Call method.

        Args:
            rgb_imgs (numpy.ndarray): RGB images to be converted.

        Returns:
            numpy.ndarray, converted HSV images.
        """
        return util.rgb_to_hsvs(rgb_imgs, self.is_hwc)


class HsvToRgb(py_transforms.PyTensorOperation):
    """
    Convert one or more numpy.ndarray images from HSV to RGB.

    Args:
        is_hwc (bool): Whether the image is in shape of (H, W, C) or (N, H, W, C), otherwise
            in shape of (C, H, W) or (N, C, H, W) (default=False).

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

    @check_hsv_to_rgb
    def __init__(self, is_hwc=False):
        self.is_hwc = is_hwc
        self.random = False

    def __call__(self, hsv_imgs):
        """
        Call method.

        Args:
            hsv_imgs (numpy.ndarray): HSV images to be converted.

        Returns:
            numpy.ndarray, converted RGB images.
        """
        return util.hsv_to_rgbs(hsv_imgs, self.is_hwc)


class RandomColor(py_transforms.PyTensorOperation):
    """
    Adjust the color balance of the input PIL Image by a random degree.

    Args:
        degrees (sequence): Range of color adjustment degree to be randomly chosen from,
            which should be in shape of (min, max) (default=(0.1,1.9)).

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
            img (PIL Image): Image to be color adjusted.

        Returns:
            PIL Image, color adjusted image.
        """

        return util.random_color(img, self.degrees)


class RandomSharpness(py_transforms.PyTensorOperation):
    """
    Adjust the sharpness of the input PIL Image by a random degree.

    Args:
        degrees (sequence): Range of sharpness adjustment degree to be randomly chosen from, which
            should be in shape of (min, max) (default=(0.1,1.9)).
            Degree of 0.0 gives a blurred image, degree of 1.0 gives the original image,
            and degree of 2.0 increases the sharpness by a factor of 2.

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
            img (PIL Image): Image to be sharpness adjusted.

        Returns:
            PIL Image, sharpness adjusted image.
        """

        return util.random_sharpness(img, self.degrees)


class AdjustGamma(py_transforms.PyTensorOperation):
    """
    Perform gamma correction on the input PIL Image.

    Args:
        gamma (float): Gamma parameter in the correction equation, which must be non negative.
        gain (float, optional): The constant multiplier (default=1.0).

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.AdjustGamma(),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_adjust_gamma
    def __init__(self, gamma, gain=1.0):
        self.gamma = gamma
        self.gain = gain
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL Image): Image to be gamma adjusted.

        Returns:
            PIL Image, gamma adjusted image.
        """

        return util.adjust_gamma(img, self.gamma, self.gain)


class AutoContrast(py_transforms.PyTensorOperation):
    """
    Automatically maximize the contrast of the input PIL Image.

    Args:
        cutoff (float, optional): Percent of pixels to be cut off from the histogram,
            which must be in range of [0.0, 50.0) (default=0.0).
        ignore (Union[int, sequence], optional): Pixel values to be ignored (default=None).

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
            img (PIL Image): Image to be automatically contrasted.

        Returns:
            PIL Image, automatically contrasted image.
        """

        return util.auto_contrast(img, self.cutoff, self.ignore)


class Invert(py_transforms.PyTensorOperation):
    """
    Invert the colors of the input PIL Image.

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
            img (PIL Image): Image to be color inverted.

        Returns:
            PIL Image, color inverted image.
        """

        return util.invert_color(img)


class Equalize(py_transforms.PyTensorOperation):
    """
    Apply histogram equalization on the input PIL Image.

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
            img (PIL Image): Image to be equalized.

        Returns:
            PIL Image, equalized image.
        """

        return util.equalize(img)


class UniformAugment(py_transforms.PyTensorOperation):
    """
    Uniformly select a number of transformations from a sequence and apply them
    sequentially and randomly, which means that there is a chance that a chosen
    transformation will not be applied.

    All transformations in the sequence require the output type to be the same as
    the input. Thus, the latter one can deal with the output of the previous one.

    Args:
         transforms (sequence): Sequence of transformations to be chosen from.
         num_ops (int, optional): Number of transformations to be sequentially and randomly applied (default=2).

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
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image, transformed image.
        """
        return util.uniform_augment(img, self.transforms.copy(), self.num_ops)
