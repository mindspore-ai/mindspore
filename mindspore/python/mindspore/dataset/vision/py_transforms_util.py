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
from .utils import Inter, FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM, PERSPECTIVE, AFFINE

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
        dtype (str): Output datatype of normalize, only worked when pad_channel is True. Default: "float32".

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
        raise ValueError("{0}\n: Failed to decode given image.".format(e))
    except AttributeError as e:
        raise ValueError("{0}\n: Failed to decode, Image might already be decoded.".format(e))


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
        output_type: The datatype of the NumPy output. e.g. np.float32

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
        interpolation (interpolation mode): Image interpolation mode. Default is Inter.BILINEAR = 2.

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
        interpolation (interpolation mode): Image interpolation mode. Default is Inter.BILINEAR = 2.
        max_attempts (int): The maximum number of attempts to propose a valid crop_area. Default: 10.
            If exceeded, fall back to use center_crop instead.

    Returns:
        PIL.Image.Image, randomly cropped and resized image.
    """
    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))
    if isinstance(size, int):
        size = (size, size)
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        size = size
    else:
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
            Default: None.
        pad_if_needed (bool): Pad the image if either side is smaller than
            the given output size. Default: False.
        fill_value (Union[int, tuple]): The pixel intensity of the borders if
            the padding_mode is 'constant'. If it is a 3-tuple, it is used to
            fill R, G, B channels respectively.
        padding_mode (str): The method of padding. Can be any of ['constant', 'edge', 'reflect', 'symmetric'].

              - 'constant', means it fills the border with constant values
              - 'edge', means it pads with the last value on the edge
              - 'reflect', means it reflects the values on the edge omitting the last
                value of edge
              - 'symmetric', means it reflects the values on the edge repeating the last
                value of edge

    Returns:
        PIL.Image.Image, cropped image.
    """
    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))
    if isinstance(size, int):
        size = (size, size)
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        size = size
    else:
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
            the brightness is adjusted. 0 gives a black image, 1 gives the original.

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
            the contrast is adjusted. 0 gives a solid gray image, 1 gives the original.

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
            the saturation is adjusted. 0 will give a black and white image, 1 will
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
            0 means no shift that gives the original image while both -0.5 and 0.5
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
    except Exception:
        raise RuntimeError("output_type: " + str(output_type) + " is not a valid datatype.")


def rotate(img, angle, resample, expand, center, fill_value):
    """
    Rotate the input PIL Image by angle.

    Args:
        img (PIL.Image.Image): Image to be rotated.
        angle (int or float): Rotation angle in degrees, counter-clockwise.
        resample (Union[Inter.NEAREST, Inter.BILINEAR, Inter.BICUBIC], optional): An optional resampling filter.
            If omitted, or if the image has mode "1" or "P", it is set to be Inter.NEAREST.
        expand (bool, optional):  Optional expansion flag. If set to True, expand the output
            image to make it large enough to hold the entire rotated image.
            If set to False or omitted, make the output image the same size as the input.
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


def random_lighting(img, alpha):
    """
    Add AlexNet-style PCA-based noise to an image.

    Args:
        img (PIL.Image.Image): Image to be added AlexNet-style PCA-based noise.
        alpha (float, optional): Intensity of the image.

    Returns:
        PIL.Image.Image, image with noise added.
    """
    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))
    if img.mode != 'RGB':
        img = img.convert("RGB")

    alpha_r = np.random.normal(loc=0.0, scale=alpha)
    alpha_g = np.random.normal(loc=0.0, scale=alpha)
    alpha_b = np.random.normal(loc=0.0, scale=alpha)
    table = np.array([
        [55.46 * -0.5675, 4.794 * 0.7192, 1.148 * 0.4009],
        [55.46 * -0.5808, 4.794 * -0.0045, 1.148 * -0.8140],
        [55.46 * -0.5836, 4.794 * -0.6948, 1.148 * 0.4203]
    ])
    pca_r = table[0][0] * alpha_r + table[0][1] * alpha_g + table[0][2] * alpha_b
    pca_g = table[1][0] * alpha_r + table[1][1] * alpha_g + table[1][2] * alpha_b
    pca_b = table[2][0] * alpha_r + table[2][1] * alpha_g + table[2][2] * alpha_b
    img_arr = np.array(img).astype(np.float64)
    img_arr[:, :, 0] += pca_r
    img_arr[:, :, 1] += pca_g
    img_arr[:, :, 2] += pca_b
    img_arr = np.uint8(np.minimum(np.maximum(img_arr, 0), 255))
    img = Image.fromarray(img_arr)
    return img


def random_rotation(img, degrees, resample, expand, center, fill_value):
    """
    Rotate the input PIL Image by a random angle.

    See <https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.rotate>.

    Args:
        img (PIL.Image.Image): Image to be rotated.
        degrees (Union[int, float, sequence]): Range of random rotation degrees.
            If `degrees` is a number, the range will be converted to (-degrees, degrees).
            If `degrees` is a sequence, it should be (min, max).
        resample (Union[Inter.NEAREST, Inter.BILINEAR, Inter.BICUBIC], optional): An optional resampling filter.
            If omitted, or if the image has mode "1" or "P", it is set to be Inter.NEAREST.
        expand (bool, optional):  Optional expansion flag. If set to True, expand the output
            image to make it large enough to hold the entire rotated image.
            If set to False or omitted, make the output image the same size as the input.
            Note that the expand flag assumes rotation around the center and no translation.
        center (tuple, optional): Optional center of rotation (a 2-tuple).
            Origin is the top left corner.
        fill_value (Union[int, tuple]): Optional fill color for the area outside the rotated image.
            If it is a 3-tuple, it is used for R, G, B channels respectively.
            If it is an integer, it is used for all RGB channels.

    Returns:
        PIL.Image.Image, Rotated image.
    """
    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))

    if isinstance(degrees, numbers.Number):
        if degrees < 0:
            raise ValueError("If degrees is a single number, it cannot be negative.")
        degrees = (-degrees, degrees)
    elif isinstance(degrees, (list, tuple)):
        if len(degrees) != 2:
            raise ValueError("If degrees is a sequence, the length must be 2.")
    else:
        raise TypeError("Degrees must be a single non-negative number or a sequence.")

    if isinstance(fill_value, list):
        # Ensure fill_value of type list (from serialize JSON support) is converted to type tuple
        fill_value = tuple(fill_value)

    angle = random.uniform(degrees[0], degrees[1])
    return rotate(img, angle, resample, expand, center, fill_value)


def five_crop(img, size):
    """
    Generate 5 cropped images (one central and four corners).

    Args:
        img (PIL.Image.Image): PIL Image to be cropped.
        size (Union[int, sequence]): The output size of the crop.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).

    Returns:
            img_tuple (tuple), a tuple of 5 PIL Image
                (top_left, top_right, bottom_left, bottom_right, center).
    """
    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))

    if isinstance(size, int):
        size = (size, size)
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        size = size
    else:
        raise TypeError("Size should be a single number or a list/tuple (h, w) of length 2.")

    # PIL.Image.Image.size returns in (width, height) order
    img_width, img_height = img.size
    crop_height, crop_width = size
    if crop_height > img_height or crop_width > img_width:
        raise ValueError("Crop size {} is larger than input image size {}.".format(size, (img_height, img_width)))
    center = center_crop(img, (crop_height, crop_width))
    top_left = img.crop((0, 0, crop_width, crop_height))
    top_right = img.crop((img_width - crop_width, 0, img_width, crop_height))
    bottom_left = img.crop((0, img_height - crop_height, crop_width, img_height))
    bottom_right = img.crop((img_width - crop_width, img_height - crop_height, img_width, img_height))

    return top_left, top_right, bottom_left, bottom_right, center


def ten_crop(img, size, use_vertical_flip=False):
    """
    Generate 10 cropped images (first 5 from FiveCrop, second 5 from their flipped version).

    The default is horizontal flipping, use_vertical_flip=False.

    Args:
        img (PIL.Image.Image): PIL Image to be cropped.
        size (Union[int, sequence]): The output size of the crop.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).
        use_vertical_flip (bool): Flip the image vertically instead of horizontally if set to True.

    Returns:
        tuple[PIL.Image.Image], a tuple of 10 PIL Image
            (top_left, top_right, bottom_left, bottom_right, center) of original image +
            (top_left, top_right, bottom_left, bottom_right, center) of flipped image.
    """
    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))

    if isinstance(size, int):
        size = (size, size)
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        size = size
    else:
        raise TypeError("Size should be a single number or a list/tuple (h, w) of length 2.")

    first_five_crop = five_crop(img, size)

    if use_vertical_flip:
        img = vertical_flip(img)
    else:
        img = horizontal_flip(img)

    second_five_crop = five_crop(img, size)

    return first_five_crop + second_five_crop


def grayscale(img, num_output_channels):
    """
    Convert the input PIL Image to grayscale image.

    Args:
        img (PIL.Image.Image): PIL Image to be converted to grayscale.
        num_output_channels (int): Number of channels of the output grayscale image (1 or 3).

    Returns:
        PIL.Image.Image, grayscaled image.
    """
    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))

    if num_output_channels == 1:
        img = img.convert('L')
    elif num_output_channels == 3:
        # each channel is the same grayscale layer
        img = img.convert('L')
        np_gray = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_gray, np_gray, np_gray])
        img = Image.fromarray(np_img, 'RGB')
    else:
        raise ValueError('num_output_channels should be either 1 or 3. Got {}.'.format(num_output_channels))

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
            Default: None.
        fill_value (Union[int, tuple]): The pixel intensity of the borders if
            the padding_mode is "constant". If it is a 3-tuple, it is used to
            fill R, G, B channels respectively.
        padding_mode (str): The method of padding. Can be any of ['constant', 'edge', 'reflect', 'symmetric'].

              - 'constant', means it fills the border with constant values
              - 'edge', means it pads with the last value on the edge
              - 'reflect', means it reflects the values on the edge omitting the last
                value of edge
              - 'symmetric', means it reflects the values on the edge repeating the last
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


def get_perspective_params(img, distortion_scale):
    """Helper function to get parameters for RandomPerspective.
    """
    img_width, img_height = img.size
    distorted_half_width = int(img_width / 2 * distortion_scale)
    distorted_half_height = int(img_height / 2 * distortion_scale)
    top_left = (random.randint(0, distorted_half_width),
                random.randint(0, distorted_half_height))
    top_right = (random.randint(img_width - distorted_half_width - 1, img_width - 1),
                 random.randint(0, distorted_half_height))
    bottom_right = (random.randint(img_width - distorted_half_width - 1, img_width - 1),
                    random.randint(img_height - distorted_half_height - 1, img_height - 1))
    bottom_left = (random.randint(0, distorted_half_width),
                   random.randint(img_height - distorted_half_height - 1, img_height - 1))

    start_points = [(0, 0), (img_width - 1, 0), (img_width - 1, img_height - 1), (0, img_height - 1)]
    end_points = [top_left, top_right, bottom_right, bottom_left]
    return start_points, end_points


def perspective(img, start_points, end_points, interpolation=Inter.BICUBIC):
    """
    Apply perspective transformation to the input PIL Image.

    Args:
        img (PIL.Image.Image): PIL Image to be applied perspective transformation.
        start_points (list): List of [top_left, top_right, bottom_right, bottom_left] of the original image.
        end_points: List of [top_left, top_right, bottom_right, bottom_left] of the transformed image.
        interpolation (interpolation mode): Image interpolation mode, Default is Inter.BICUBIC = 3.

    Returns:
        PIL.Image.Image, image after being perspectively transformed.
    """

    def _input_to_coeffs(original_points, transformed_points):
        # Get the coefficients (a, b, c, d, e, f, g, h) for the perspective transforms.
        # According to "Using Projective Geometry to Correct a Camera" from AMS.
        # http://www.ams.org/publicoutreach/feature-column/fc-2013-03
        # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Geometry.c#L377

        matrix = []
        for pt1, pt2 in zip(transformed_points, original_points):
            matrix.append([pt1[0], pt1[1], 1, 0, 0, 0, -pt2[0] * pt1[0], -pt2[0] * pt1[1]])
            matrix.append([0, 0, 0, pt1[0], pt1[1], 1, -pt2[1] * pt1[0], -pt2[1] * pt1[1]])
        matrix_a = np.array(matrix, dtype=np.float)
        matrix_b = np.array(original_points, dtype=np.float).reshape(8)
        res = np.linalg.lstsq(matrix_a, matrix_b, rcond=None)[0]
        return res.tolist()

    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))

    coeffs = _input_to_coeffs(start_points, end_points)
    return img.transform(img.size, PERSPECTIVE, coeffs, interpolation)


def get_erase_params(np_img, scale, ratio, value, bounded, max_attempts):
    """Helper function to get parameters for RandomErasing/Cutout.
    """
    if not is_numpy(np_img):
        raise TypeError('img should be NumPy array. Got {}.'.format(type(np_img)))

    image_c, image_h, image_w = np_img.shape
    area = image_h * image_w

    for _ in range(max_attempts):
        erase_area = random.uniform(scale[0], scale[1]) * area
        aspect_ratio = random.uniform(ratio[0], ratio[1])
        erase_w = int(round(math.sqrt(erase_area * aspect_ratio)))
        erase_h = int(round(erase_w / aspect_ratio))
        erase_shape = (image_c, erase_h, erase_w)

        if erase_h < image_h and erase_w < image_w:
            if bounded:
                i = random.randint(0, image_h - erase_h)
                j = random.randint(0, image_w - erase_w)
            else:
                def clip(x, lower, upper):
                    return max(lower, min(x, upper))

                x = random.randint(0, image_w)
                y = random.randint(0, image_h)
                x1 = clip(x - erase_w // 2, 0, image_w)
                x2 = clip(x + erase_w // 2, 0, image_w)
                y1 = clip(y - erase_h // 2, 0, image_h)
                y2 = clip(y + erase_h // 2, 0, image_h)

                i, j, erase_h, erase_w = y1, x1, y2 - y1, x2 - x1

            if isinstance(value, numbers.Number):
                erase_value = value
            elif isinstance(value, (str, bytes)):
                erase_value = np.random.normal(loc=0.0, scale=1.0, size=erase_shape)
            elif isinstance(value, (tuple, list)) and len(value) == 3:
                value = np.array(value)
                erase_value = np.multiply(np.ones(erase_shape), value[:, None, None])
            else:
                raise ValueError("The value for erasing should be either a single value, or a string "
                                 "'random', or a sequence of 3 elements for RGB respectively.")

            return i, j, erase_h, erase_w, erase_value

    # exceeding max_attempts, return original image
    return 0, 0, image_h, image_w, np_img


def erase(np_img, i, j, height, width, erase_value, inplace=False):
    """
    Erase the pixels, within a selected rectangle region, to the given value. Applied on the input NumPy image array.

    Args:
        np_img (numpy.ndarray): NumPy image array of shape (C, H, W) to be erased.
        i (int): The height component of the top left corner (height, width).
        j (int): The width component of the top left corner (height, width).
        height (int): Height of the erased region.
        width (int): Width of the erased region.
        erase_value: Erase value return from helper function get_erase_params().
        inplace (bool, optional): Apply this transform inplace. Default: False.

    Returns:
        np_img (numpy.ndarray), Erased NumPy image array.
    """
    if not is_numpy(np_img):
        raise TypeError('img should be NumPy array. Got {}.'.format(type(np_img)))

    if not inplace:
        np_img = np_img.copy()
    # (i, j) here are the coordinates of axes (height, width) as in CHW
    np_img[:, i:i + height, j:j + width] = erase_value
    return np_img


def linear_transform(np_img, transformation_matrix, mean_vector):
    """
    Apply linear transformation to the input NumPy image array, given a square transformation matrix and a mean_vector.

    The transformation first flattens the input array and subtract mean_vector from it, then computes the
    dot product with the transformation matrix, and reshapes it back to its original shape.

    Args:
        np_img (numpy.ndarray): NumPy image array of shape (C, H, W) to be linear transformed.
        transformation_matrix (numpy.ndarray): a square transformation matrix of shape (D, D), D = C x H x W.
        mean_vector (numpy.ndarray): a NumPy ndarray of shape (D,) where D = C x H x W.

    Returns:
        np_img (numpy.ndarray), Linear transformed image.
    """
    if not is_numpy(np_img):
        raise TypeError('img should be NumPy array. Got {}'.format(type(np_img)))
    if transformation_matrix.shape[0] != transformation_matrix.shape[1]:
        raise ValueError("transformation_matrix should be a square matrix. "
                         "Got shape {} instead".format(transformation_matrix.shape))
    if np.prod(np_img.shape) != transformation_matrix.shape[0]:
        raise ValueError("transformation_matrix shape {0} not compatible with "
                         "Numpy image shape {1}.".format(transformation_matrix.shape, np_img.shape))
    if mean_vector.shape[0] != transformation_matrix.shape[0]:
        raise ValueError("mean_vector length {0} should match either one dimension of the square "
                         "transformation_matrix {1}.".format(mean_vector.shape[0], transformation_matrix.shape))
    zero_centered_img = np_img.reshape(1, -1) - mean_vector
    transformed_img = np.dot(zero_centered_img, transformation_matrix)
    if transformed_img.size != np_img.size:
        raise ValueError("Linear transform failed, input shape should match with transformation_matrix.")
    transformed_img = transformed_img.reshape(np_img.shape)
    return transformed_img


def random_affine(img, angle, translations, scale, shear, resample, fill_value=0):
    """
    Applies a random Affine transformation on the input PIL Image.

    Args:
        img (PIL.Image.Image): Image to be applied affine transformation.
        angle (Sequence): Rotation angle in degrees, clockwise.
        translations (Sequence): Translations in horizontal and vertical axis.
        scale (Sequence): Scale parameter.
        shear (Sequence): Shear amount parallel to X axis and Y axis.
        resample (Inter): Resampling filter.
        fill_value (Union[tuple, int], optional): Optional fill_value to fill the area outside the transform
            in the output image. Used only in Pillow versions > 5.0.0.
            If None, no filling is performed.

    Returns:
        PIL.Image.Image, randomly affine transformed image.

    """
    if not is_pil(img):
        raise ValueError("Input image should be a Pillow image.")

    # rotation
    angle = random.uniform(angle[0], angle[1])

    # translation
    if translations is not None:
        max_dx = translations[0] * img.size[0]
        max_dy = translations[1] * img.size[1]
        translations = (np.round(random.uniform(-max_dx, max_dx)),
                        np.round(random.uniform(-max_dy, max_dy)))
    else:
        translations = (0, 0)

    # scale
    if scale is not None:
        scale = random.uniform(scale[0], scale[1])
    else:
        scale = 1.0

    # shear
    if shear is not None:
        if len(shear) == 2:
            shear = [random.uniform(shear[0], shear[1]), 0.]
        elif len(shear) == 4:
            shear = [random.uniform(shear[0], shear[1]),
                     random.uniform(shear[2], shear[3])]
    else:
        shear = 0.0

    output_size = img.size
    center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)

    angle = math.radians(angle)
    if isinstance(shear, (tuple, list)) and len(shear) == 2:
        shear = [math.radians(s) for s in shear]
    elif isinstance(shear, numbers.Number):
        shear = math.radians(shear)
        shear = [shear, 0]
    else:
        raise ValueError(
            "Shear should be a single value or a tuple/list containing " +
            "two values. Got {}.".format(shear))

    scale = 1.0 / scale

    # Inverted rotation matrix with scale and shear
    d = math.cos(angle + shear[0]) * math.cos(angle + shear[1]) + \
        math.sin(angle + shear[0]) * math.sin(angle + shear[1])
    matrix = [
        math.cos(angle + shear[0]), math.sin(angle + shear[0]), 0,
        -math.sin(angle + shear[1]), math.cos(angle + shear[1]), 0
    ]
    matrix = [scale / d * m for m in matrix]

    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    matrix[2] += matrix[0] * (-center[0] - translations[0]) + matrix[1] * (-center[1] - translations[1])
    matrix[5] += matrix[3] * (-center[0] - translations[0]) + matrix[4] * (-center[1] - translations[1])

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += center[0]
    matrix[5] += center[1]

    # Ensure fill_value of type list (from serialize JSON support) is converted to type tuple
    kwarg_fill_value = tuple(fill_value) if isinstance(fill_value, list) else fill_value

    if Image.__version__ >= '5':
        kwargs = {"fillcolor": kwarg_fill_value}
    else:
        kwargs = {}
    return img.transform(output_size, AFFINE, matrix, resample, **kwargs)


def mix_up_single(batch_size, img, label, alpha=0.2):
    """
    Apply mix up transformation to image and label in single batch internal, One hot encoding should done before this.

    Args:
        batch_size (int): The batch size of dataset.
        img (numpy.ndarray): NumPy image to be applied mix up transformation.
        label (numpy.ndarray): NumPy label to be applied mix up transformation.
        alpha (float): The mix up rate.

    Returns:
        mix_img (numpy.ndarray): NumPy image after being applied mix up transformation.
        mix_label (numpy.ndarray): NumPy label after being applied mix up transformation.
    """

    def cir_shift(data):
        index = list(range(1, batch_size)) + [0]
        data = data[index, ...]
        return data

    lam = np.random.beta(alpha, alpha, batch_size)
    lam_img = lam.reshape((batch_size, 1, 1, 1))
    mix_img = lam_img * img + (1 - lam_img) * cir_shift(img)

    lam_label = lam.reshape((batch_size, 1))
    mix_label = lam_label * label + (1 - lam_label) * cir_shift(label)

    return mix_img, mix_label


def mix_up_muti(tmp, batch_size, img, label, alpha=0.2):
    """
    Apply mix up transformation to image and label in continuous batch, one hot encoding should done before this.

    Args:
        tmp (class object): mainly for saving the tmp parameter.
        batch_size (int): the batch size of dataset.
        img (numpy.ndarray): NumPy image to be applied mix up transformation.
        label (numpy.ndarray): NumPy label to be applied mix up transformation.
        alpha (float):  refer to the mix up rate.

    Returns:
        mix_img (numpy.ndarray): NumPy image after being applied mix up transformation.
        mix_label (numpy.ndarray): NumPy label after being applied mix up transformation.
    """
    lam = np.random.beta(alpha, alpha, batch_size)
    if tmp.is_first:
        lam = np.ones(batch_size)
        tmp.is_first = False

    lam_img = lam.reshape((batch_size, 1, 1, 1))
    mix_img = lam_img * img + (1 - lam_img) * tmp.image

    lam_label = lam.reshape(batch_size, 1)
    mix_label = lam_label * label + (1 - lam_label) * tmp.label
    tmp.image = mix_img
    tmp.label = mix_label

    return mix_img, mix_label


def rgb_to_bgr(np_rgb_img, is_hwc):
    """
    Convert RGB img to BGR img.

    Args:
        np_rgb_img (numpy.ndarray): NumPy RGB image array of shape (H, W, C) or (C, H, W) to be converted.
        is_hwc (Bool): If True, the shape of np_hsv_img is (H, W, C), otherwise must be (C, H, W).

    Returns:
        np_bgr_img (numpy.ndarray), NumPy BGR image with same type of np_rgb_img.
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
        is_hwc (Bool): If True, the shape of np_hsv_img is (H, W, C), otherwise must be (C, H, W).

    Returns:
        np_hsv_img (numpy.ndarray), NumPy HSV image with same type of np_rgb_img.
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
        is_hwc (Bool): If True, the shape of np_rgb_imgs is (H, W, C) or (N, H, W, C);
                       If False, the shape of np_rgb_imgs is (C, H, W) or (N, C, H, W).

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
        is_hwc (Bool): If True, the shape of np_hsv_img is (H, W, C), otherwise must be (C, H, W).

    Returns:
        np_rgb_img (numpy.ndarray), NumPy HSV image with same shape of np_hsv_img.
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
        is_hwc (Bool): If True, the shape of np_hsv_imgs is (H, W, C) or (N, H, W, C);
                       If False, the shape of np_hsv_imgs is (C, H, W) or (N, C, H, W).

    Returns:
        np_rgb_imgs (numpy.ndarray), NumPy RGB images with same type of np_hsv_imgs.
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


def random_color(img, degrees):
    """
    Adjust the color of the input PIL Image by a random degree.

    Args:
        img (PIL.Image.Image): Image to be color adjusted.
        degrees (sequence): Range of random color adjustment degrees.
            It should be in (min, max) format. Default: (0.1,1.9).

    Returns:
        PIL.Image.Image, color adjusted image.
    """

    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))

    v = (degrees[1] - degrees[0]) * random.random() + degrees[0]
    return ImageEnhance.Color(img).enhance(v)


def random_sharpness(img, degrees):
    """
    Adjust the sharpness of the input PIL Image by a random degree.

    Args:
        img (PIL.Image.Image): Image to be sharpness adjusted.
        degrees (sequence): Range of random sharpness adjustment degrees.
            It should be in (min, max) format. Default: (0.1,1.9).

    Returns:
        PIL.Image.Image, sharpness adjusted image.
    """

    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))

    v = (degrees[1] - degrees[0]) * random.random() + degrees[0]
    return ImageEnhance.Sharpness(img).enhance(v)


def adjust_gamma(img, gamma, gain):
    """
    Adjust gamma of the input PIL Image.

    Args:
        img (PIL.Image.Image): Image to be augmented with AdjustGamma.
        gamma (float): Non negative real number, same as gamma in the equation.
        gain (float, optional): The constant multiplier.

    Returns:
        PIL.Image.Image, augmented image.

    """

    if not is_pil(img):
        raise TypeError("img should be PIL image. Got {}.".format(type(img)))

    gamma_table = [(255 + 1 - 1e-3) * gain * pow(x / 255., gamma) for x in range(256)]
    if len(img.split()) == 3:
        gamma_table = gamma_table * 3
        img = img.point(gamma_table)
    elif len(img.split()) == 1:
        img = img.point(gamma_table)
    return img


def auto_contrast(img, cutoff, ignore):
    """
    Automatically maximize the contrast of the input PIL Image.

    Args:
        img (PIL.Image): Image to be augmented with AutoContrast.
        cutoff (float, optional): Percent of pixels to cut off from the histogram. Default: 0.0.
        ignore (Union[int, Sequence[int]], optional): Pixel values to ignore. Default: None.

    Returns:
        PIL.Image, augmented image.
    """

    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))

    return ImageOps.autocontrast(img, cutoff, ignore)


def invert_color(img):
    """
    Invert colors of input PIL Image.

    Args:
        img (PIL.Image.Image): Image to be color inverted.

    Returns:
        PIL.Image.Image, color inverted image.

    """

    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))

    return ImageOps.invert(img)


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


def uniform_augment(img, transforms, num_ops):
    """
    Uniformly select and apply a number of transforms sequentially from
    a list of transforms. Randomly assigns a probability to each transform for
    each image to decide whether apply it or not.
    All the transforms in transform list must have the same input/output data type.

    Args:
        img: Image to be applied transformation.
        transforms (list): List of transformations to be chosen from to apply.
        num_ops (int): number of transforms to sequentially aaply.

    Returns:
        img, Transformed image.

    """

    op_idx = np.random.choice(len(transforms), size=num_ops, replace=False)
    for idx in op_idx:
        augment_op = transforms[idx]
        pr = random.random()
        if random.random() < pr:
            img = augment_op(img.copy())

    return img
