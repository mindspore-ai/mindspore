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
"""Built-in py_transforms_utils functions.
"""
import colorsys
import io
import math
import numbers
import random

import numpy as np
from PIL import Image, ImageOps, ImageEnhance

from ..core.py_util_helpers import is_numpy
from .utils import (
    Inter,
    FLIP_LEFT_RIGHT,
    FLIP_TOP_BOTTOM,
)

augment_error_message = "img should be PIL image. Got {}. Use Decode() for encoded data or ToPIL() for decoded data."


def is_pil(img):
    """
    Check if the input image is PIL format.

    Args:
        img: Image to be checked.

    Returns:
        bool, True if input is PIL.Image.Image.
    """
    return isinstance(img, Image.Image)


def normalize(img, mean, std, pad_channel=False, dtype="float32"):
    """
    Normalize the image between [0, 1] with respect to mean and standard deviation.

    Args:
        img (numpy.ndarray): Image array of shape CHW to be normalized.
        mean (list): List of mean values for each channel, w.r.t channel order.
        std (list): List of standard deviations for each channel, w.r.t. channel order.
        pad_channel (bool): Whether to pad a extra channel with value zero.
        dtype (str): Output datatype of normalize, only worked when `pad_channel` is ``True``.
            Default: ``"float32"``.

    Returns:
        img (numpy.ndarray), Normalized image.
    """
    if not is_numpy(img):
        raise TypeError("img should be NumPy image. Got {}.".format(type(img)))

    if img.ndim != 3:
        raise TypeError('img dimension should be 3. Got {}.'.format(img.ndim))

    if np.issubdtype(img.dtype, np.integer):
        raise NotImplementedError("Unsupported image datatype: [{}], pls execute [ToTensor] before [Normalize]."
                                  .format(img.dtype))

    num_channels = img.shape[0]  # shape is (C, H, W)

    if len(mean) != len(std):
        raise ValueError("Length of mean and std must be equal.")
    # if length equal to 1, adjust the mean and std arrays to have the correct
    # number of channels (replicate the values)
    if len(mean) == 1:
        mean = [mean[0]] * num_channels
        std = [std[0]] * num_channels
    elif len(mean) != num_channels:
        raise ValueError("Length of mean and std must both be 1 or equal to the number of channels({0})."
                         .format(num_channels))

    mean = np.array(mean, dtype=img.dtype)
    std = np.array(std, dtype=img.dtype)

    image = (img - mean[:, None, None]) / std[:, None, None]
    if pad_channel:
        zeros = np.zeros([1, image.shape[1], image.shape[2]], dtype=np.float32)
        image = np.concatenate((image, zeros), axis=0)
        if dtype == "float16":
            image = image.astype(np.float16)
    return image


def decode(img):
    """
    Decode the input image to PIL Image format in RGB mode.

    Args:
        img: Image to be decoded.

    Returns:
        img (PIL.Image.Image), Decoded image in RGB mode.
    """

    try:
        data = io.BytesIO(img)
        img = Image.open(data)
        return img.convert('RGB')
    except IOError as e:
        raise ValueError("{0}\n: Failed to decode given image.".format(e)) from e
    except AttributeError as e:
        raise ValueError("{0}\n: Failed to decode, Image might already be decoded.".format(e)) from e


def hwc_to_chw(img):
    """
    Transpose the input image from shape (H, W, C) to (C, H, W).
    If the input image is of shape <H, W>, it will remain unchanged.

    Args:
        img (numpy.ndarray): Image to be converted.

    Returns:
        img (numpy.ndarray), Converted image.
    """
    if not is_numpy(img):
        raise TypeError('img should be NumPy array. Got {}.'.format(type(img)))
    if img.ndim not in (2, 3):
        raise TypeError("img dimension should be 2 or 3. Got {}.".format(img.ndim))
    if img.ndim == 2:
        return img
    return img.transpose(2, 0, 1).copy()


def to_tensor(img, output_type):
    """
    Change the input image (PIL.Image.Image or numpy.ndarray) to numpy.ndarray format.

    Args:
        img (Union[PIL.Image.Image, numpy.ndarray]): Image to be converted.
        output_type: The datatype of the NumPy output. e.g. ``np.float32``.

    Returns:
        img (numpy.ndarray), Converted image.
    """
    if not (is_pil(img) or is_numpy(img)):
        raise TypeError("The input image should be of type numpy.ndarray or PIL.Image.Image. Got {}.".format(type(img)))

    img = np.asarray(img)
    if img.ndim not in (2, 3):
        raise TypeError("The dimension of input image should be 2 or 3. Got {}.".format(img.ndim))

    if img.ndim == 2:
        img = img[:, :, None]

    img = hwc_to_chw(img)

    img = img / 255.
    return to_type(img, output_type)


def to_pil(img):
    """
    Convert the input image to PIL format.

    Args:
        img: Image to be converted.

    Returns:
        img (PIL.Image.Image), Converted image.
    """
    if not is_pil(img):
        if not isinstance(img, np.ndarray):
            raise TypeError("The input image should be of type numpy.ndarray or PIL.Image.Image. "
                            "Got {}.".format(type(img)))
        if img.ndim not in (2, 3):
            raise ValueError("The dimension of input image should be 2 or 3. Got {}.".format(img.ndim))
        if img.ndim == 2:
            img = np.expand_dims(img, 2)
        if img.shape[-1] > 4:
            raise ValueError("The channel of input image should not exceed 4. Got {}.".format(img.shape[-1]))
        if img.shape[-1] == 1:
            if img.dtype not in (np.bool_, np.int8, np.int16, np.int32, np.uint8, np.uint16, np.uint32, np.float32,
                                 np.float64):
                raise TypeError("The input image type {} is not supported when "
                                "image shape is [H, W] or [H, W, 1].".format(img.dtype))
            img = img[:, :, 0]
        elif img.dtype != np.uint8:
            raise TypeError("The input image type {} is not supported when "
                            "image shape is [H, W, 2], [H, W, 3] or [H, W, 4].".format(img.dtype))
        return Image.fromarray(img)
    return img


def horizontal_flip(img):
    """
    Flip the input image horizontally.

    Args:
        img (PIL.Image.Image): Image to be flipped horizontally.

    Returns:
        PIL.Image.Image, Horizontally flipped image.
    """
    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))

    return img.transpose(FLIP_LEFT_RIGHT)


def vertical_flip(img):
    """
    Flip the input image vertically.

    Args:
        img (PIL.Image.Image): Image to be flipped vertically.

    Returns:
        PIL.Image.Image, Vertically flipped image.
    """
    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))

    return img.transpose(FLIP_TOP_BOTTOM)


def random_horizontal_flip(img, prob):
    """
    Randomly flip the input image horizontally.

    Args:
        img (PIL.Image.Image): Image to be flipped.
            If the given probability is above the random probability, then the image is flipped.
        prob (float): Probability of the image being flipped.

    Returns:
        PIL.Image.Image, Converted image.
    """
    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))

    if prob > random.random():
        img = horizontal_flip(img)
    return img


def random_vertical_flip(img, prob):
    """
    Randomly flip the input image vertically.

    Args:
        img (PIL.Image.Image): Image to be flipped.
            If the given probability is above the random probability, then the image is flipped.
        prob (float): Probability of the image being flipped.

    Returns:
        PIL.Image.Image, Converted image.
    """
    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))

    if prob > random.random():
        img = vertical_flip(img)
    return img


def crop(img, top, left, height, width):
    """
    Crop the input PIL Image.

    Args:
        img (PIL.Image.Image): Image to be cropped. (0,0) denotes the top left corner of the image,
            in the directions of (width, height).
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.

    Returns:
        PIL.Image.Image, cropped image.
    """
    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))

    return img.crop((left, top, left + width, top + height))


def resize(img, size, interpolation=Inter.BILINEAR):
    """
    Resize the input PIL Image to desired size.

    Args:
        img (PIL.Image.Image): Image to be resized.
        size (Union[int, sequence]): The output size of the resized image.
            If size is an integer, smaller edge of the image will be resized to this value with
            the same image aspect ratio.
            If size is a sequence of (height, width), this will be the desired output size.
        interpolation (interpolation mode): Image interpolation mode. Default: ``Inter.BILINEAR = 2``.

    Returns:
        PIL.Image.Image, resized image.
    """
    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, (list, tuple)) and len(size) == 2)):
        raise TypeError('Size should be a single number or a list/tuple (h, w) of length 2.'
                        'Got {}.'.format(size))

    if isinstance(size, int):
        img_width, img_height = img.size
        aspect_ratio = img_width / img_height  # maintain the aspect ratio
        if (img_width <= img_height and img_width == size) or \
                (img_height <= img_width and img_height == size):
            return img
        if img_width < img_height:
            out_width = size
            out_height = int(size / aspect_ratio)
            return img.resize((out_width, out_height), interpolation)
        out_height = size
        out_width = int(size * aspect_ratio)
        return img.resize((out_width, out_height), interpolation)
    return img.resize(size[::-1], interpolation)


def center_crop(img, size):
    """
    Crop the input PIL Image at the center to the given size.

    Args:
        img (PIL.Image.Image): Image to be cropped.
        size (Union[int, tuple]): The size of the crop box.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).

    Returns:
        PIL.Image.Image, cropped image.
    """
    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))

    if isinstance(size, int):
        size = (size, size)
    img_width, img_height = img.size
    crop_height, crop_width = size
    crop_top = int(round((img_height - crop_height) / 2.))
    crop_left = int(round((img_width - crop_width) / 2.))
    return crop(img, crop_top, crop_left, crop_height, crop_width)


def random_resize_crop(img, size, scale, ratio, interpolation=Inter.BILINEAR, max_attempts=10):
    """
    Crop the input PIL Image to a random size and aspect ratio.

    Args:
        img (PIL.Image.Image): Image to be randomly cropped and resized.
        size (Union[int, sequence]): The size of the output image.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).
        scale (tuple): Range (min, max) of respective size of the original size to be cropped.
        ratio (tuple): Range (min, max) of aspect ratio to be cropped.
        interpolation (interpolation mode): Image interpolation mode. Default: ``Inter.BILINEAR = 2``.
        max_attempts (int): The maximum number of attempts to propose a valid crop_area. Default: ``10``.
            If exceeded, fall back to use center_crop instead.

    Returns:
        PIL.Image.Image, randomly cropped and resized image.
    """
    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))
    if isinstance(size, int):
        size = (size, size)
    elif not (isinstance(size, (tuple, list)) and len(size) == 2):
        raise TypeError("Size should be a single integer or a list/tuple (h, w) of length 2.")

    if scale[0] > scale[1] or ratio[0] > ratio[1]:
        raise ValueError("Range should be in the order of (min, max).")

    def _input_to_factor(img, scale, ratio):
        img_width, img_height = img.size
        img_area = img_width * img_height

        for _ in range(max_attempts):
            crop_area = random.uniform(scale[0], scale[1]) * img_area
            # in case of non-symmetrical aspect ratios,
            # use uniform distribution on a logarithmic scale.
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            width = int(round(math.sqrt(crop_area * aspect_ratio)))
            height = int(round(width / aspect_ratio))

            if 0 < width <= img_width and 0 < height <= img_height:
                top = random.randint(0, img_height - height)
                left = random.randint(0, img_width - width)
                return top, left, height, width

        # exceeding max_attempts, use center crop
        img_ratio = img_width / img_height
        if img_ratio < ratio[0]:
            width = img_width
            height = int(round(width / ratio[0]))
        elif img_ratio > ratio[1]:
            height = img_height
            width = int(round(height * ratio[1]))
        else:
            width = img_width
            height = img_height
        top = int(round((img_height - height) / 2.))
        left = int(round((img_width - width) / 2.))
        return top, left, height, width

    top, left, height, width = _input_to_factor(img, scale, ratio)
    img = crop(img, top, left, height, width)
    img = resize(img, size, interpolation)
    return img


def random_crop(img, size, padding, pad_if_needed, fill_value, padding_mode):
    """
    Crop the input PIL Image at a random location.

    Args:
        img (PIL.Image.Image): Image to be randomly cropped.
        size (Union[int, sequence]): The output size of the cropped image.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).
        padding (Union[int, sequence], optional): The number of pixels to pad the image.
            If a single number is provided, it pads all borders with this value.
            If a tuple or lists of 2 values are provided, it pads the (left and top)
            with the first value and (right and bottom) with the second value.
            If 4 values are provided as a list or tuple,
            it pads the left, top, right and bottom respectively.
            Default: ``None``.
        pad_if_needed (bool): Pad the image if either side is smaller than
            the given output size. Default: ``False``.
        fill_value (Union[int, tuple]): The pixel intensity of the borders if
            the `padding_mode` is ``'constant'``. If it is a 3-tuple, it is used to
            fill R, G, B channels respectively.
        padding_mode (str): The method of padding. Can be ``'constant'``, ``'edge'``,
            ``'reflect'``, ``'symmetric'``.

            - ``'constant'``, means it fills the border with constant values
            - ``'edge'``, means it pads with the last value on the edge
            - ``'reflect'``, means it reflects the values on the edge omitting the last
              value of edge
            - ``'symmetric'``, means it reflects the values on the edge repeating the last
              value of edge

    Returns:
        PIL.Image.Image, cropped image.
    """
    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))
    if isinstance(size, int):
        size = (size, size)
    elif not (isinstance(size, (tuple, list)) and len(size) == 2):
        raise TypeError("Size should be a single integer or a list/tuple (h, w) of length 2.")

    if isinstance(fill_value, list):
        # Ensure fill_value of type list (from serialize JSON support) is converted to type tuple
        fill_value = tuple(fill_value)

    def _input_to_factor(img, size):
        img_width, img_height = img.size
        height, width = size
        if height > img_height or width > img_width:
            raise ValueError("Crop size {} is larger than input image size {}.".format(size, (img_height, img_width)))

        if width == img_width and height == img_height:
            return 0, 0, img_height, img_width

        top = random.randint(0, img_height - height)
        left = random.randint(0, img_width - width)
        return top, left, height, width

    if padding is not None:
        img = pad(img, padding, fill_value, padding_mode)
    # pad width when needed, img.size (width, height), crop size (height, width)
    if pad_if_needed and img.size[0] < size[1]:
        img = pad(img, (size[1] - img.size[0], 0), fill_value, padding_mode)
    # pad height when needed
    if pad_if_needed and img.size[1] < size[0]:
        img = pad(img, (0, size[0] - img.size[1]), fill_value, padding_mode)

    top, left, height, width = _input_to_factor(img, size)
    return crop(img, top, left, height, width)


def adjust_brightness(img, brightness_factor):
    """
    Adjust brightness of an image.

    Args:
        img (PIL.Image.Image): Image to be adjusted.
        brightness_factor (float): A non negative number indicated the factor by which
            the brightness is adjusted. ``0`` gives a black image, ``1`` gives the original.

    Returns:
        PIL.Image.Image, brightness adjusted image.
    """
    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img, contrast_factor):
    """
    Adjust contrast of an image.

    Args:
        img (PIL.Image.Image): PIL Image to be adjusted.
        contrast_factor (float): A non negative number indicated the factor by which
            the contrast is adjusted. ``0`` gives a solid gray image, ``1`` gives the original.

    Returns:
        PIL.Image.Image, contrast adjusted image.
    """
    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(img, saturation_factor):
    """
    Adjust saturation of an image.

    Args:
        img (PIL.Image.Image): PIL Image to be adjusted.
        saturation_factor (float):  A non negative number indicated the factor by which
            the saturation is adjusted. ``0`` will give a black and white image, ``1`` will
            give the original.

    Returns:
        PIL.Image.Image, saturation adjusted image.
    """
    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_hue(img, hue_factor):
    """
    Adjust hue of an image. The Hue is changed by changing the HSV values after image is converted to HSV.

    Args:
        img (PIL.Image.Image): PIL Image to be adjusted.
        hue_factor (float):  Amount to shift the Hue channel. Value should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel. This
            is because Hue wraps around when rotated 360 degrees.
            ``0`` means no shift that gives the original image while both -0.5 and 0.5
            will give an image with complementary colors .

    Returns:
        PIL.Image.Image, hue adjusted image.
    """
    image = img
    image_hue_factor = hue_factor
    if not -0.5 <= image_hue_factor <= 0.5:
        raise ValueError('image_hue_factor {} is not in [-0.5, 0.5].'.format(image_hue_factor))

    if not is_pil(image):
        raise TypeError(augment_error_message.format(type(image)))

    mode = image.mode
    if mode in {'L', '1', 'I', 'F'}:
        return image

    hue, saturation, value = img.convert('HSV').split()

    np_hue = np.array(hue, dtype=np.uint8)

    with np.errstate(over='ignore'):
        np_hue += np.uint8(image_hue_factor * 255)
    hue = Image.fromarray(np_hue, 'L')

    image = Image.merge('HSV', (hue, saturation, value)).convert(mode)
    return image


def to_type(img, output_type):
    """
    Convert the NumPy image array to desired NumPy dtype.

    Args:
        img (numpy): NumPy image to cast to desired NumPy dtype.
        output_type (Numpy datatype): NumPy dtype to cast to.

    Returns:
        img (numpy.ndarray), Converted image.
    """
    if not is_numpy(img):
        raise TypeError("img should be NumPy image. Got {}.".format(type(img)))

    try:
        return img.astype(output_type)
    except Exception as exc:
        raise RuntimeError("output_type: " + str(output_type) + " is not a valid datatype.") from exc


def rotate(img, angle, resample, expand, center, fill_value):
    """
    Rotate the input PIL Image by angle.

    Args:
        img (PIL.Image.Image): Image to be rotated.
        angle (int or float): Rotation angle in degrees, counter-clockwise.
        resample (Union[Inter.NEAREST, Inter.BILINEAR, Inter.BICUBIC], optional): An optional resampling filter.
            If omitted, or if the image has mode "1" or "P", it is set to be ``Inter.NEAREST``.
        expand (bool, optional):  Optional expansion flag. If set to ``True``, expand the output
            image to make it large enough to hold the entire rotated image.
            If set to ``False`` or omitted, make the output image the same size as the input.
            Note that the expand flag assumes rotation around the center and no translation.
        center (tuple, optional): Optional center of rotation (a 2-tuple).
            Origin is the top left corner.
        fill_value (Union[int, tuple]): Optional fill color for the area outside the rotated image.
            If it is a 3-tuple, it is used for R, G, B channels respectively.
            If it is an integer, it is used for all RGB channels.

    Returns:
        PIL.Image.Image, rotated image.

    https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.rotate
    """
    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))

    if isinstance(fill_value, int):
        fill_value = tuple([fill_value] * 3)
    elif isinstance(fill_value, list):
        # Ensure fill_value of type list (from serialize JSON support) is converted to type tuple
        fill_value = tuple(fill_value)

    angle = angle % 360.0
    if resample == Inter.ANTIALIAS:
        if angle not in [0, 90, 180, 270] or center:
            raise ValueError("When using Inter.ANTIALIAS, center needs to be None and "
                             "angle needs to be an integer multiple of 90.")
    return img.rotate(angle, resample, expand, center, fillcolor=fill_value)


def random_color_adjust(img, brightness, contrast, saturation, hue):
    """
    Randomly adjust the brightness, contrast, saturation, and hue of an image.

    Args:
        img (PIL.Image.Image): Image to have its color adjusted randomly.
        brightness (Union[float, tuple]): Brightness adjustment factor. Cannot be negative.
            If it is a float, the factor is uniformly chosen from the range [max(0, 1-brightness), 1+brightness].
            If it is a sequence, it should be [min, max] for the range.
        contrast (Union[float, tuple]): Contrast adjustment factor. Cannot be negative.
            If it is a float, the factor is uniformly chosen from the range [max(0, 1-contrast), 1+contrast].
            If it is a sequence, it should be [min, max] for the range.
        saturation (Union[float, tuple]): Saturation adjustment factor. Cannot be negative.
            If it is a float, the factor is uniformly chosen from the range [max(0, 1-saturation), 1+saturation].
            If it is a sequence, it should be [min, max] for the range.
        hue (Union[float, tuple]): Hue adjustment factor.
            If it is a float, the range will be [-hue, hue]. Value should be 0 <= hue <= 0.5.
            If it is a sequence, it should be [min, max] where -0.5 <= min <= max <= 0.5.

    Returns:
        PIL.Image.Image, image after random adjustment of its color.
    """
    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))

    def _input_to_factor(value, input_name, center=1, bound=(0, float('inf')), non_negative=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("The input value of {} cannot be negative.".format(input_name))
            # convert value into a range
            value = [center - value, center + value]
            if non_negative:
                value[0] = max(0, value[0])
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("Please check your value range of {} is valid and "
                                 "within the bound {}.".format(input_name, bound))
        else:
            raise TypeError("Input of {} should be either a single value, or a list/tuple of "
                            "length 2.".format(input_name))
        factor = random.uniform(value[0], value[1])
        return factor

    brightness_factor = _input_to_factor(brightness, 'brightness')
    contrast_factor = _input_to_factor(contrast, 'contrast')
    saturation_factor = _input_to_factor(saturation, 'saturation')
    hue_factor = _input_to_factor(hue, 'hue', center=0, bound=(-0.5, 0.5), non_negative=False)

    transforms = [lambda img: adjust_brightness(img, brightness_factor),
                  lambda img: adjust_contrast(img, contrast_factor),
                  lambda img: adjust_saturation(img, saturation_factor),
                  lambda img: adjust_hue(img, hue_factor)]

    # apply color adjustments in a random order
    random.shuffle(transforms)
    for transform in transforms:
        img = transform(img)

    return img




def pad(img, padding, fill_value, padding_mode):
    """
    Pad the image according to padding parameters.

    Args:
        img (PIL.Image.Image): Image to be padded.
        padding (Union[int, sequence], optional): The number of pixels to pad the image.
            If a single number is provided, it pads all borders with this value.
            If a tuple or lists of 2 values are provided, it pads the (left and top)
            with the first value and (right and bottom) with the second value.
            If 4 values are provided as a list or tuple,
            it pads the left, top, right and bottom respectively.
            Default: ``None``.
        fill_value (Union[int, tuple]): The pixel intensity of the borders if
            the `padding_mode` is ``"constant"``. If it is a 3-tuple, it is used to
            fill R, G, B channels respectively.
        padding_mode (str): The method of padding. Can be ``'constant'``, ``'edge'``, ``'reflect'``, ``'symmetric'``.

              - ``'constant'``, means it fills the border with constant values
              - ``'edge'``, means it pads with the last value on the edge
              - ``'reflect'``, means it reflects the values on the edge omitting the last
                value of edge
              - ``'symmetric'``, means it reflects the values on the edge repeating the last
                value of edge

    Returns:
        PIL.Image.Image, padded image.
    """
    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))

    if isinstance(padding, numbers.Number):
        top = bottom = left = right = padding

    elif isinstance(padding, (tuple, list)):
        if len(padding) == 2:
            left = right = padding[0]
            top = bottom = padding[1]
        elif len(padding) == 4:
            left = padding[0]
            top = padding[1]
            right = padding[2]
            bottom = padding[3]
        else:
            raise ValueError("The size of the padding list or tuple should be 2 or 4.")
    else:
        raise TypeError("Padding can be any of: a number, a tuple or list of size 2 or 4.")

    if isinstance(fill_value, list):
        # Ensure fill_value of type list (from serialize JSON support) is converted to type tuple
        fill_value = tuple(fill_value)

    if not isinstance(fill_value, (numbers.Number, str, tuple)):
        raise TypeError("fill_value can be any of: an integer, a string or a tuple.")

    if padding_mode not in ['constant', 'edge', 'reflect', 'symmetric']:
        raise ValueError("Padding mode should be 'constant', 'edge', 'reflect', or 'symmetric'.")

    if padding_mode == 'constant':
        if img.mode == 'P':
            palette = img.getpalette()
            image = ImageOps.expand(img, border=(left, top, right, bottom), fill=fill_value)
            image.putpalette(palette)
            return image
        if isinstance(fill_value, tuple) and (img.mode == 'L' or img.mode == '1'):
            fill_value = (fill_value[0],)
        return ImageOps.expand(img, border=(left, top, right, bottom), fill=fill_value)

    if img.mode == 'P':
        palette = img.getpalette()
        img = np.asarray(img)
        img = np.pad(img, ((top, bottom), (left, right)), padding_mode)
        img = Image.fromarray(img)
        img.putpalette(palette)
        return img

    img = np.asarray(img)
    if len(img.shape) == 3:
        img = np.pad(img, ((top, bottom), (left, right), (0, 0)), padding_mode)
    if len(img.shape) == 2:
        img = np.pad(img, ((top, bottom), (left, right)), padding_mode)

    return Image.fromarray(img)












def rgb_to_bgr(np_rgb_img, is_hwc):
    """
    Convert RGB img to BGR img.

    Args:
        np_rgb_img (numpy.ndarray): NumPy RGB image array of shape (H, W, C) or (C, H, W) to be converted.
        is_hwc (Bool): If ``True``, the shape of np_hsv_img is (H, W, C), otherwise must be (C, H, W).

    Returns:
        np_bgr_img (numpy.ndarray), NumPy BGR image with same type of `np_rgb_img`.
    """
    if is_hwc:
        np_bgr_img = np_rgb_img[:, :, ::-1]
    else:
        np_bgr_img = np_rgb_img[::-1, :, :]
    return np_bgr_img


def rgb_to_bgrs(np_rgb_imgs, is_hwc):
    """
    Convert RGB imgs to BGR imgs.

    Args:
        np_rgb_imgs (numpy.ndarray): NumPy RGB images array of shape (H, W, C) or (N, H, W, C),
                                      or (C, H, W) or (N, C, H, W) to be converted.
        is_hwc (Bool): If True, the shape of np_rgb_imgs is (H, W, C) or (N, H, W, C);
                       If False, the shape of np_rgb_imgs is (C, H, W) or (N, C, H, W).

    Returns:
        np_bgr_imgs (numpy.ndarray), NumPy BGR images with same type of np_rgb_imgs.
    """
    if not is_numpy(np_rgb_imgs):
        raise TypeError("img should be NumPy image. Got {}".format(type(np_rgb_imgs)))

    if not isinstance(is_hwc, bool):
        raise TypeError("is_hwc should be bool type. Got {}.".format(type(is_hwc)))

    shape_size = len(np_rgb_imgs.shape)

    if shape_size not in (3, 4):
        raise TypeError("img shape should be (H, W, C)/(N, H, W, C)/(C ,H, W)/(N, C, H, W). "
                        "Got {}.".format(np_rgb_imgs.shape))

    if shape_size == 3:
        batch_size = 0
        if is_hwc:
            num_channels = np_rgb_imgs.shape[2]
        else:
            num_channels = np_rgb_imgs.shape[0]
    else:
        batch_size = np_rgb_imgs.shape[0]
        if is_hwc:
            num_channels = np_rgb_imgs.shape[3]
        else:
            num_channels = np_rgb_imgs.shape[1]

    if num_channels != 3:
        raise TypeError("img should be 3 channels RGB img. Got {} channels.".format(num_channels))
    if batch_size == 0:
        return rgb_to_bgr(np_rgb_imgs, is_hwc)
    return np.array([rgb_to_bgr(img, is_hwc) for img in np_rgb_imgs])


def rgb_to_hsv(np_rgb_img, is_hwc):
    """
    Convert RGB img to HSV img.

    Args:
        np_rgb_img (numpy.ndarray): NumPy RGB image array of shape (H, W, C) or (C, H, W) to be converted.
        is_hwc (Bool): If ``True``, the shape of np_hsv_img is (H, W, C), otherwise must be (C, H, W).

    Returns:
        np_hsv_img (numpy.ndarray), NumPy HSV image with same type of `np_rgb_img`.
    """
    if is_hwc:
        r, g, b = np_rgb_img[:, :, 0], np_rgb_img[:, :, 1], np_rgb_img[:, :, 2]
    else:
        r, g, b = np_rgb_img[0, :, :], np_rgb_img[1, :, :], np_rgb_img[2, :, :]
    to_hsv = np.vectorize(colorsys.rgb_to_hsv)
    h, s, v = to_hsv(r, g, b)
    if is_hwc:
        axis = 2
    else:
        axis = 0
    np_hsv_img = np.stack((h, s, v), axis=axis)
    return np_hsv_img


def rgb_to_hsvs(np_rgb_imgs, is_hwc):
    """
    Convert RGB imgs to HSV imgs.

    Args:
        np_rgb_imgs (numpy.ndarray): NumPy RGB images array of shape (H, W, C) or (N, H, W, C),
                                      or (C, H, W) or (N, C, H, W) to be converted.
        is_hwc (Bool): If ``True``, the shape of `np_rgb_imgs` is (H, W, C) or (N, H, W, C);
                       If ``False``, the shape of `np_rgb_imgs` is (C, H, W) or (N, C, H, W).

    Returns:
        np_hsv_imgs (numpy.ndarray), NumPy HSV images with same type of np_rgb_imgs.
    """
    if not is_numpy(np_rgb_imgs):
        raise TypeError("img should be NumPy image. Got {}".format(type(np_rgb_imgs)))

    if not isinstance(is_hwc, bool):
        raise TypeError("is_hwc should be bool type. Got {}.".format(type(is_hwc)))

    shape_size = len(np_rgb_imgs.shape)

    if shape_size not in (3, 4):
        raise TypeError("img shape should be (H, W, C)/(N, H, W, C)/(C ,H, W)/(N, C, H, W). "
                        "Got {}.".format(np_rgb_imgs.shape))

    if shape_size == 3:
        batch_size = 0
        if is_hwc:
            num_channels = np_rgb_imgs.shape[2]
        else:
            num_channels = np_rgb_imgs.shape[0]
    else:
        batch_size = np_rgb_imgs.shape[0]
        if is_hwc:
            num_channels = np_rgb_imgs.shape[3]
        else:
            num_channels = np_rgb_imgs.shape[1]

    if num_channels != 3:
        raise TypeError("img should be 3 channels RGB img. Got {} channels.".format(num_channels))
    if batch_size == 0:
        return rgb_to_hsv(np_rgb_imgs, is_hwc)
    return np.array([rgb_to_hsv(img, is_hwc) for img in np_rgb_imgs])


def hsv_to_rgb(np_hsv_img, is_hwc):
    """
    Convert HSV img to RGB img.

    Args:
        np_hsv_img (numpy.ndarray): NumPy HSV image array of shape (H, W, C) or (C, H, W) to be converted.
        is_hwc (Bool): If ``True``, the shape of `np_hsv_img` is (H, W, C), otherwise must be (C, H, W).

    Returns:
        np_rgb_img (numpy.ndarray), NumPy HSV image with same shape of `np_hsv_img`.
    """
    if is_hwc:
        h, s, v = np_hsv_img[:, :, 0], np_hsv_img[:, :, 1], np_hsv_img[:, :, 2]
    else:
        h, s, v = np_hsv_img[0, :, :], np_hsv_img[1, :, :], np_hsv_img[2, :, :]
    to_rgb = np.vectorize(colorsys.hsv_to_rgb)
    r, g, b = to_rgb(h, s, v)

    if is_hwc:
        axis = 2
    else:
        axis = 0
    np_rgb_img = np.stack((r, g, b), axis=axis)
    return np_rgb_img


def hsv_to_rgbs(np_hsv_imgs, is_hwc):
    """
    Convert HSV imgs to RGB imgs.

    Args:
        np_hsv_imgs (numpy.ndarray): NumPy HSV images array of shape (H, W, C) or (N, H, W, C),
                                      or (C, H, W) or (N, C, H, W) to be converted.
        is_hwc (Bool): If ``True``, the shape of `np_hsv_imgs` is (H, W, C) or (N, H, W, C);
                       If ``False``, the shape of `np_hsv_imgs` is (C, H, W) or (N, C, H, W).

    Returns:
        np_rgb_imgs (numpy.ndarray), NumPy RGB images with same type of `np_hsv_imgs`.
    """
    if not is_numpy(np_hsv_imgs):
        raise TypeError("img should be NumPy image. Got {}.".format(type(np_hsv_imgs)))

    if not isinstance(is_hwc, bool):
        raise TypeError("is_hwc should be bool type. Got {}.".format(type(is_hwc)))

    shape_size = len(np_hsv_imgs.shape)

    if shape_size not in (3, 4):
        raise TypeError("img shape should be (H, W, C)/(N, H, W, C)/(C, H, W)/(N, C, H, W). "
                        "Got {}.".format(np_hsv_imgs.shape))

    if shape_size == 3:
        batch_size = 0
        if is_hwc:
            num_channels = np_hsv_imgs.shape[2]
        else:
            num_channels = np_hsv_imgs.shape[0]
    else:
        batch_size = np_hsv_imgs.shape[0]
        if is_hwc:
            num_channels = np_hsv_imgs.shape[3]
        else:
            num_channels = np_hsv_imgs.shape[1]

    if num_channels != 3:
        raise TypeError("img should be 3 channels RGB img. Got {} channels.".format(num_channels))
    if batch_size == 0:
        return hsv_to_rgb(np_hsv_imgs, is_hwc)
    return np.array([hsv_to_rgb(img, is_hwc) for img in np_hsv_imgs])


def equalize(img):
    """
    Equalize the histogram of input PIL Image.

    Args:
        img (PIL.Image.Image): Image to be equalized

    Returns:
        PIL.Image.Image, equalized image.

    """

    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))

    return ImageOps.equalize(img)
