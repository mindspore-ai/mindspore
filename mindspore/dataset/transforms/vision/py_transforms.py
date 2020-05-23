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
The module vision.py_transforms is implemented basing on python
PIL. This module provides many kinds of image augmentations. It also provides
transferring methods between PIL Image and numpy array. For users who prefer
python PIL in image learning task, this module is a good tool to process image
augmentations. Users could also self-define their own augmentations with python
PIL.
"""
import numbers
import random

import numpy as np
from PIL import Image

from . import py_transforms_util as util
from .validators import check_prob, check_crop, check_resize_interpolation, check_random_resize_crop, \
    check_normalize_py, check_random_crop, check_random_color_adjust, check_random_rotation, \
    check_transforms_list, check_random_apply, check_ten_crop, check_num_channels, check_pad, \
    check_random_perspective, check_random_erasing, check_cutout, check_linear_transform, check_random_affine, \
    check_mix_up
from .utils import Inter, Border

DE_PY_INTER_MODE = {Inter.NEAREST: Image.NEAREST,
                    Inter.LINEAR: Image.LINEAR,
                    Inter.CUBIC: Image.CUBIC}

DE_PY_BORDER_TYPE = {Border.CONSTANT: 'constant',
                     Border.EDGE: 'edge',
                     Border.REFLECT: 'reflect',
                     Border.SYMMETRIC: 'symmetric'}


class ComposeOp:
    """
    Compose a list of transforms.

    .. Note::
        ComposeOp takes a list of transformations either provided in py_transforms or from user-defined implementation;
        each can be an initialized transformation class or a lambda function, as long as the output from the last
        transformation is a single tensor of type numpy.ndarray. See below for an example of how to use ComposeOp
        with py_transforms classes and check out FiveCrop or TenCrop for the use of them in conjunction with lambda
        functions.

    Args:
        transforms (list): List of transformations to be applied.

    Examples:
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.transforms.vision.py_transforms as py_transforms
        >>> dataset_dir = "path/to/imagefolder_directory"
        >>> # create a dataset that reads all files in dataset_dir with 8 threads
        >>> dataset = ds.ImageFolderDatasetV2(dataset_dir, num_parallel_workers=8)
        >>> # create a list of transformations to be applied to the image data
        >>> transform = py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                                      py_transforms.RandomHorizontalFlip(0.5),
        >>>                                      py_transforms.ToTensor(),
        >>>                                      py_transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
        >>>                                      py_transforms.RandomErasing()])
        >>> # apply the transform to the dataset through dataset.map()
        >>> dataset = dataset.map(input_columns="image", operations=transform())
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self):
        """
        Call method.

        Returns:
            lambda function, Lambda function that takes in an img to apply transformations on.
        """
        return lambda img: util.compose(img, self.transforms)


class ToTensor:
    """
    Convert the input Numpy image array or PIL image of shape (H,W,C) to a Numpy ndarray of shape (C,H,W).

    Note:
        The ranges of values in height and width dimension changes from [0, 255] to [0.0, 1.0]. Type cast to output_type
        (default Numpy float 32).
        The range of channel dimension remains the same.

    Args:
        output_type (numpy datatype, optional): The datatype of the numpy output (default=np.float32).

    Examples:
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.RandomHorizontalFlip(0.5),
        >>>                          py_transforms.ToTensor()])
    """

    def __init__(self, output_type=np.float32):
        self.output_type = output_type

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL Image): PIL Image to be converted to numpy.ndarray.

        Returns:
            img (numpy.ndarray), Converted image.
        """
        return util.to_tensor(img, self.output_type)


class ToType:
    """
    Convert the input Numpy image array to desired numpy dtype.

    Args:
        output_type (numpy datatype): The datatype of the numpy output. e.g. np.float32.

    Examples:
        >>> import numpy as np
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.RandomHorizontalFlip(0.5),
        >>>                          py_transforms.ToTensor(),
        >>>                          py_transforms.ToType(np.float32)])
    """

    def __init__(self, output_type):
        self.output_type = output_type

    def __call__(self, img):
        """
        Call method.

        Args:
            numpy object : numpy object to be type swapped.

        Returns:
            img (numpy.ndarray), Converted image.
        """
        return util.to_type(img, self.output_type)


class HWC2CHW:
    """
    Transpose a Numpy image array; shape (H, W, C) to shape (C, H, W).
    """

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
    Convert the input decoded Numpy image array of RGB mode to a PIL Image of RGB mode.

    Examples:
        >>> # data is already decoded, but not in PIL Image format
        >>> py_transforms.ComposeOp([py_transforms.ToPIL(),
        >>>                          py_transforms.RandomHorizontalFlip(0.5),
        >>>                          py_transforms.ToTensor()])
    """

    def __call__(self, img):
        """
        Call method.

        Args:
            img (numpy.ndarray): Decoded image array, of RGB mode, to be converted to PIL Image.

        Returns:
            img (PIL Image), Image converted to PIL Image of RGB mode.
        """
        return util.to_pil(img)


class Decode:
    """
    Decode the input image to PIL Image format in RGB mode.

    Examples:
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.RandomHorizontalFlip(0.5),
        >>>                          py_transforms.ToTensor()])
    """

    def __call__(self, img):
        """
        Call method.

        Args:
            img (Bytes-like Objects):Image to be decoded.

        Returns:
            img (PIL Image), Decoded image in RGB mode.
        """
        return util.decode(img)


class Normalize:
    """
    Normalize the input Numpy image array of shape (C, H, W) with the given mean and standard deviation.

    The values of the array need to be in range [0.0, 1.0].

    Args:
        mean (sequence): List or tuple of mean values for each channel, w.r.t channel order.
        std (sequence): List or tuple of standard deviations for each channel, w.r.t. channel order.

    Examples:
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.RandomHorizontalFlip(0.5),
        >>>                          py_transforms.ToTensor(),
        >>>                          py_transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262))])
    """

    @check_normalize_py
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Call method.

        Args:
            img (numpy.ndarray): Image array to be normalized.

        Returns:
            img (numpy.ndarray), Normalized Image array.
        """
        return util.normalize(img, self.mean, self.std)


class RandomCrop:
    """
    Crop the input PIL Image at a random location.

    Args:
        size (int or sequence): The output size of the cropped image.
            If size is an int, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).
        padding (int or sequence, optional): The number of pixels to pad the image (default=None).
            If padding is not None, pad image firstly with padding values.
            If a single number is provided, it pads all borders with this value.
            If a tuple or list of 2 values are provided, it pads the (left and top)
            with the first value and (right and bottom) with the second value.
            If 4 values are provided as a list or tuple,
            it pads the left, top, right and bottom respectively.
        pad_if_needed (bool, optional): Pad the image if either side is smaller than
            the given output size (default=False).
        fill_value (int or tuple, optional): filling value (default=0).
            The pixel intensity of the borders if the padding_mode is Border.CONSTANT.
            If it is a 3-tuple, it is used to fill R, G, B channels respectively.
        padding_mode (str, optional): The method of padding (default=Border.CONSTANT). Can be any of
            [Border.CONSTANT, Border.EDGE, Border.REFLECT, Border.SYMMETRIC].

            - Border.CONSTANT, means it fills the border with constant values.

            - Border.EDGE, means it pads with the last value on the edge.

            - Border.REFLECT, means it reflects the values on the edge omitting the last
              value of edge.

            - Border.SYMMETRIC, means it reflects the values on the edge repeating the last
              value of edge.

    Examples:
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.RandomCrop(224),
        >>>                          py_transforms.ToTensor()])
    """

    @check_random_crop
    def __init__(self, size, padding=None, pad_if_needed=False, fill_value=0, padding_mode=Border.CONSTANT):
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
            PIL Image, Cropped image.
        """
        return util.random_crop(img, self.size, self.padding, self.pad_if_needed,
                                self.fill_value, self.padding_mode)


class RandomHorizontalFlip:
    """
    Randomly flip the input image horizontally with a given probability.

    Args:
        prob (float, optional): Probability of the image being flipped (default=0.5).

    Examples:
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.RandomHorizontalFlip(0.5),
        >>>                          py_transforms.ToTensor()])
    """

    @check_prob
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL Image): Image to be flipped horizontally.

        Returns:
            img (PIL Image), Randomly flipped image.
        """
        return util.random_horizontal_flip(img, self.prob)


class RandomVerticalFlip:
    """
    Randomly flip the input image vertically with a given probability.

    Args:
        prob (float, optional): Probability of the image being flipped (default=0.5).

    Examples:
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.RandomVerticalFlip(0.5),
        >>>                          py_transforms.ToTensor()])
    """

    @check_prob
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL Image): Image to be flipped vertically.

        Returns:
            img (PIL Image), Randomly flipped image.
        """
        return util.random_vertical_flip(img, self.prob)


class Resize:
    """
    Resize the input PIL Image to the given size.

    Args:
        size (int or sequence): The output size of the resized image.
            If size is an int, smaller edge of the image will be resized to this value with
            the same image aspect ratio.
            If size is a sequence of length 2, it should be (height, width).
        interpolation (Inter mode, optional): Image interpolation mode (default=Inter.BILINEAR).
            It can be any of [Inter.BILINEAR, Inter.NEAREST, Inter.BICUBIC].

            - Inter.BILINEAR, means interpolation method is bilinear interpolation.

            - Inter.NEAREST, means interpolation method is nearest-neighbor interpolation.

            - Inter.BICUBIC, means interpolation method is bicubic interpolation.

    Examples:
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.Resize(256),
        >>>                          py_transforms.ToTensor()])
    """

    @check_resize_interpolation
    def __init__(self, size, interpolation=Inter.BILINEAR):
        self.size = size
        self.interpolation = DE_PY_INTER_MODE[interpolation]

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL Image): Image to be resized.

        Returns:
            img (PIL Image), Resize image.
        """
        return util.resize(img, self.size, self.interpolation)


class RandomResizedCrop:
    """
    Extract crop from the input image and resize it to a random size and aspect ratio.

    Args:
        size (int or sequence): The size of the output image.
            If size is an int, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).
        scale (tuple, optional): Range (min, max) of respective size of the original size
            to be cropped (default=(0.08, 1.0)).
        ratio (tuple, optional): Range (min, max) of aspect ratio to be cropped (default=(3. / 4., 4. / 3.)).
        interpolation (Inter mode, optional): Image interpolation mode (default=Inter.BILINEAR).
            It can be any of [Inter.BILINEAR, Inter.NEAREST, Inter.BICUBIC].

            - Inter.BILINEAR, means interpolation method is bilinear interpolation.

            - Inter.NEAREST, means interpolation method is nearest-neighbor interpolation.

            - Inter.BICUBIC, means interpolation method is bicubic interpolation.

        max_attempts (int, optional): The maximum number of attempts to propose a valid
            crop_area (default=10). If exceeded, fall back to use center_crop instead.

    Examples:
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.RandomResizedCrop(224),
        >>>                          py_transforms.ToTensor()])
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
            img (PIL Image), Randomly cropped and resized image.
        """
        return util.random_resize_crop(img, self.size, self.scale, self.ratio,
                                       self.interpolation, self.max_attempts)


class CenterCrop:
    """
    Crop the central reigion of the input PIL Image to the given size.

    Args:
        size (int or sequence): The output size of the cropped image.
            If size is an int, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).

    Examples:
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.CenterCrop(64),
        >>>                          py_transforms.ToTensor()])
    """

    @check_crop
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL Image): Image to be center cropped.

        Returns:
            img (PIL Image), Cropped image.
        """
        return util.center_crop(img, self.size)


class RandomColorAdjust:
    """
    Perform a random brightness, contrast, saturation, and hue adjustment on the input PIL image.

    Args:
        brightness (float or tuple, optional): Brightness adjustment factor (default=(1, 1)). Cannot be negative.
            If it is a float, the factor is uniformly chosen from the range [max(0, 1-brightness), 1+brightness].
            If it is a sequence, it should be [min, max] for the range.
        contrast (float or tuple, optional): Contrast adjustment factor (default=(1, 1)). Cannot be negative.
            If it is a float, the factor is uniformly chosen from the range [max(0, 1-contrast), 1+contrast].
            If it is a sequence, it should be [min, max] for the range.
        saturation (float or tuple, optional): Saturation adjustment factor (default=(1, 1)). Cannot be negative.
            If it is a float, the factor is uniformly chosen from the range [max(0, 1-saturation), 1+saturation].
            If it is a sequence, it should be [min, max] for the range.
        hue (float or tuple, optional): Hue adjustment factor (default=(0, 0)).
            If it is a float, the range will be [-hue, hue]. Value should be 0 <= hue <= 0.5.
            If it is a sequence, it should be [min, max] where -0.5 <= min <= max <= 0.5.

    Examples:
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.RandomColorAdjust(0.4, 0.4, 0.4, 0.1),
        >>>                          py_transforms.ToTensor()])
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
            img (PIL Image): Image to have its color adjusted randomly.

        Returns:
            img (PIL Image), Image after random adjustment of its color.
        """
        return util.random_color_adjust(img, self.brightness, self.contrast, self.saturation, self.hue)


class RandomRotation:
    """
    Rotate the input PIL image by a random angle.

    Note:
        See https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.rotate.

    Args:
        degrees (int or float or sequence): Range of random rotation degrees.
            If degrees is a number, the range will be converted to (-degrees, degrees).
            If degrees is a sequence, it should be (min, max).
        resample (Inter mode, optional): An optional resampling filter (default=Inter.NEAREST).
            If omitted, or if the image has mode "1" or "P", it is set to be Inter.NEAREST.
            It can be any of [Inter.BILINEAR, Inter.NEAREST, Inter.BICUBIC].

            - Inter.BILINEAR, means resampling method is bilinear interpolation.

            - Inter.NEAREST, means resampling method is nearest-neighbor interpolation.

            - Inter.BICUBIC, means resampling method is bicubic interpolation.

        expand (bool, optional):  Optional expansion flag (default=False). If set to True, expand the output
            image to make it large enough to hold the entire rotated image.
            If set to False or omitted, make the output image the same size as the input.
            Note that the expand flag assumes rotation around the center and no translation.
        center (tuple, optional): Optional center of rotation (a 2-tuple) (default=None).
            Origin is the top left corner. Default None sets to the center of the image.
        fill_value (int or tuple, optional): Optional fill color for the area outside the rotated
            image (default=0).
            If it is a 3-tuple, it is used for R, G, B channels respectively.
            If it is an int, it is used for all RGB channels. Default is 0.

    Examples:
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.RandomRotation(30),
        >>>                          py_transforms.ToTensor()])
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
            img (PIL Image): Image to be rotated.

        Returns:
            img (PIL Image), Rotated image.
        """
        return util.random_rotation(img, self.degrees, self.resample, self.expand, self.center, self.fill_value)


class RandomOrder:
    """
    Perform a series of transforms to the input PIL image in a random oreder.

    Args:
        transforms (list): List of the transformations to be applied.

    Examples:
        >>> transforms_list = [py_transforms.CenterCrop(64), py_transforms.RandomRotation(30)]
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.RandomOrder(transforms_list),
        >>>                          py_transforms.ToTensor()])
    """

    @check_transforms_list
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL Image): Image to be applied transformations in a random order.

        Returns:
            img (PIL Image), Transformed image.
        """
        return util.random_order(img, self.transforms)


class RandomApply:
    """
    Randomly perform a series of transforms with a given probability.

    Args:
        transforms (list): List of transformations to be applied.
        prob (float, optional): The probability to apply the transformation list (default=0.5).

    Examples:
        >>> transforms_list = [py_transforms.CenterCrop(64), py_transforms.RandomRotation(30)]
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.RandomApply(transforms_list, prob=0.6),
        >>>                          py_transforms.ToTensor()])
    """

    @check_random_apply
    def __init__(self, transforms, prob=0.5):
        self.prob = prob
        self.transforms = transforms

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL Image): Image to be randomly applied a list transformations.

        Returns:
            img (PIL Image), Transformed image.
        """
        return util.random_apply(img, self.transforms, self.prob)


class RandomChoice:
    """
    Randomly select one transform from a series of transforms and applies that on the image.

    Args:
         transforms (list): List of transformations to be chosen from to apply.

    Examples:
        >>> transforms_list = [py_transforms.CenterCrop(64), py_transforms.RandomRotation(30)]
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.RandomChoice(transforms_list),
        >>>                          py_transforms.ToTensor()])
    """

    @check_transforms_list
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL Image): Image to be applied transformation.

        Returns:
            img (PIL Image), Transformed image.
        """
        return util.random_choice(img, self.transforms)


class FiveCrop:
    """
    Generate 5 cropped images (one central and four corners).

    Args:
        size (int or sequence): The output size of the crop.
            If size is an int, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).

    Examples:
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.FiveCrop(size),
        >>>                          # 4D stack of 5 images
        >>>                          lambda images: numpy.stack([py_transforms.ToTensor()(image) for image in images])])
    """

    @check_crop
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL Image): PIL Image to be cropped.

        Returns:
            img_tuple (tuple), a tuple of 5 PIL images
                (top_left, top_right, bottom_left, bottom_right, center).
        """
        return util.five_crop(img, self.size)


class TenCrop:
    """
    Generate 10 cropped images (first 5 from FiveCrop, second 5 from their flipped version).

    Args:
        size (int or sequence): The output size of the crop.
            If size is an int, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).
        use_vertical_flip (bool, optional): Flip the image vertically instead of horizontally
            if set to True (default=False).

    Examples:
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.TenCrop(size),
        >>>                          # 4D stack of 10 images
        >>>                          lambda images: numpy.stack([py_transforms.ToTensor()(image) for image in images])])
    """

    @check_ten_crop
    def __init__(self, size, use_vertical_flip=False):
        self.size = size
        self.use_vertical_flip = use_vertical_flip

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL Image): PIL Image to be cropped.

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
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.Grayscale(3),
        >>>                          py_transforms.ToTensor()])
    """

    @check_num_channels
    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL Image): PIL image to be converted to grayscale.

        Returns:
            img (PIL Image), grayscaled image.
        """
        return util.grayscale(img, num_output_channels=self.num_output_channels)


class RandomGrayscale:
    """
    Randomly convert the input image into grayscale image with a given probability.

    Args:
        prob (float, optional): Probability of the image being converted to grayscale (default=0.1).

    Examples:
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.RandomGrayscale(0.3),
        >>>                          py_transforms.ToTensor()])
    """

    @check_prob
    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL Image): PIL image to be converted to grayscale randomly.

        Returns:
            img (PIL Image), Randomly grayscale image, same number of channels as input image.
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
        padding (int or sequence): The number of pixels to pad the image.
            If a single number is provided, it pads all borders with this value.
            If a tuple or list of 2 values are provided, it pads the (left and top)
            with the first value and (right and bottom) with the second value.
            If 4 values are provided as a list or tuple,
            it pads the left, top, right and bottom respectively.
        fill_value (int or tuple, optional): Filling value (default=0). The pixel intensity
            of the borders if the padding_mode is Border.CONSTANT.
            If it is a 3-tuple, it is used to fill R, G, B channels respectively.
        padding_mode (Border mode, optional): The method of padding (default=Border.CONSTANT).
            Can be any of [Border.CONSTANT, Border.EDGE, Border.REFLECT, Border.SYMMETRIC].

            - Border.CONSTANT, means it fills the border with constant values.

            - Border.EDGE, means it pads with the last value on the edge.

            - Border.REFLECT, means it reflects the values on the edge omitting the last
              value of edge.

            - Border.SYMMETRIC, means it reflects the values on the edge repeating the last
              value of edge.

    Examples:
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          # adds 10 pixels (default black) to each side of the border of the image
        >>>                          py_transforms.Pad(padding=10),
        >>>                          py_transforms.ToTensor()])
    """

    @check_pad
    def __init__(self, padding, fill_value=0, padding_mode=Border.CONSTANT):
        self.padding = padding
        self.fill_value = fill_value
        self.padding_mode = DE_PY_BORDER_TYPE[padding_mode]

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL Image): Image to be padded.

        Returns:
            img (PIL Image), Padded image.
        """
        return util.pad(img, self.padding, self.fill_value, self.padding_mode)


class RandomPerspective:
    """
    Randomly apply perspective transformation to the input PIL Image with a given probability.

    Args:
        distortion_scale (float, optional): The scale of distortion, float between 0 and 1 (default=0.5).
        prob (float, optional): Probability of the image being applied perspective transformation (default=0.5).
        interpolation (Inter mode, optional): Image interpolation mode (default=Inter.BICUBIC).
            It can be any of [Inter.BILINEAR, Inter.NEAREST, Inter.BICUBIC].

            - Inter.BILINEAR, means interpolation method is bilinear interpolation.

            - Inter.NEAREST, means interpolation method is nearest-neighbor interpolation.

            - Inter.BICUBIC, means interpolation method is bicubic interpolation.

    Examples:
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.RandomPerspective(prob=0.1),
        >>>                          py_transforms.ToTensor()])
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
            img (PIL Image): PIL Image to be applied perspective transformation randomly.

        Returns:
            img (PIL Image), Image after being perspectively transformed randomly.
        """
        if self.prob > random.random():
            start_points, end_points = util.get_perspective_params(img, self.distortion_scale)
            return util.perspective(img, start_points, end_points, self.interpolation)
        return img


class RandomErasing:
    """
    Erase the pixels, within a selected rectangle region, to the given value.

    Randomly applied on the input Numpy image array with a given probability.

    Zhun Zhong et al. 'Random Erasing Data Augmentation' 2017 See https://arxiv.org/pdf/1708.04896.pdf

    Args:
        prob (float, optional): Probability of applying RandomErasing (default=0.5).
        scale (sequence of floats, optional): Range of the relative erase area to the
            original image (default=(0.02, 0.33)).
        ratio (sequence of floats, optional): Range of the aspect ratio of the erase
            area (default=(0.3, 3.3)).
        value (int or sequence): Erasing value (default=0).
            If value is a single int, it is applied to all pixels to be erases.
            If value is a sequence of length 3, it is applied to R, G, B channels respectively.
            If value is a str 'random', the erase value will be obtained from a standard normal distribution.
        inplace (bool, optional): Apply this transform inplace (default=False).
        max_attempts (int, optional): The maximum number of attempts to propose a valid
            erase_area (default=10). If exceeded, return the original image.

    Examples:
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.ToTensor(),
        >>>                          py_transforms.RandomErasing(value='random')])
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
            np_img (numpy.ndarray): Numpy image array of shape (C, H, W) to be randomly erased.

        Returns:
            np_img (numpy.ndarray), Erased Numpy image array.
        """
        bounded = True
        if self.prob > random.random():
            i, j, erase_h, erase_w, erase_value = util.get_erase_params(np_img, self.scale, self.ratio,
                                                                        self.value, bounded, self.max_attempts)
            return util.erase(np_img, i, j, erase_h, erase_w, erase_value, self.inplace)
        return np_img


class Cutout:
    """
    Randomly cut (mask) out a given number of square patches from the input Numpy image array.

    Terrance DeVries and Graham W. Taylor 'Improved Regularization of Convolutional Neural Networks with Cutout' 2017
    See https://arxiv.org/pdf/1708.04552.pdf

    Args:
        length (int): The side length of each square patch.
        num_patches (int, optional): Number of patches to be cut out of an image (default=1).

    Examples:
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.ToTensor(),
        >>>                          py_transforms.Cutout(80)])
    """

    @check_cutout
    def __init__(self, length, num_patches=1):
        self.length = length
        self.num_patches = num_patches

    def __call__(self, np_img):
        """
        Call method.

        Args:
            np_img (numpy.ndarray): Numpy image array of shape (C, H, W) to be cut out.

        Returns:
            np_img (numpy.ndarray), Numpy image array with square patches cut out.
        """
        if not isinstance(np_img, np.ndarray):
            raise TypeError('img should be Numpy array. Got {}'.format(type(np_img)))
        _, image_h, image_w = np_img.shape
        scale = (self.length * self.length) / (image_h * image_w)
        bounded = False

        for _ in range(self.num_patches):
            i, j, erase_h, erase_w, erase_value = util.get_erase_params(np_img, (scale, scale), (1, 1), 0, bounded, 1)
            np_img = util.erase(np_img, i, j, erase_h, erase_w, erase_value)
        return np_img


class LinearTransformation:
    """
    Apply linear transformation to the input Numpy image array, given a square transformation matrix and
    a mean_vector.

    The transformation first flattens the input array and subtract mean_vector from it, then computes the
    dot product with the transformation matrix, and reshapes it back to its original shape.

    Args:
        transformation_matrix (numpy.ndarray): a square transformation matrix of shape (D, D), D = C x H x W.
        mean_vector (numpy.ndarray): a numpy ndarray of shape (D,) where D = C x H x W.

    Examples:
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.Resize(256),
        >>>                          py_transforms.ToTensor(),
        >>>                          py_transforms.LinearTransformation(transformation_matrix, mean_vector)])
    """

    @check_linear_transform
    def __init__(self, transformation_matrix, mean_vector):
        self.transformation_matrix = transformation_matrix
        self.mean_vector = mean_vector

    def __call__(self, np_img):
        """
        Call method.

        Args:
            np_img (numpy.ndarray): Numpy image array of shape (C, H, W) to be linear transformed.

        Returns:
            np_img (numpy.ndarray), Linear transformed image.
        """
        return util.linear_transform(np_img, self.transformation_matrix, self.mean_vector)


class RandomAffine:
    """
    Apply Random affine transformation to the input PIL image.

    Args:
        degrees (int or float or sequence): Range of the rotation degrees.
            If degrees is a number, the range will be (-degrees, degrees).
            If degrees is a sequence, it should be (min, max).
        translate (sequence, optional): Sequence (tx, ty) of maximum translation in
            x(horizontal) and y(vertical) directions (default=None).
            The horizontal and vertical shift is selected randomly from the range:
            (-tx*width, tx*width) and (-ty*height, ty*height), respectively.
            If None, no translations gets applied.
        scale (sequence, optional): Scaling factor interval (default=None, riginal scale is used).
        shear (int or float or sequence, optional): Range of shear factor (default=None).
            If a number 'shear', then a shear parallel to the x axis in the range of (-shear, +shear) is applied.
            If a tuple or list of size 2, then a shear parallel to the x axis in the range of (shear[0], shear[1])
            is applied.
            If a tuple of list of size 4, then a shear parallel to x axis in the range of (shear[0], shear[1])
            and a shear parallel to y axis in the range of (shear[2], shear[3]) is applied.
            If None, no shear is applied.
        resample (Inter mode, optional): An optional resampling filter (default=Inter.NEAREST).
            If omitted, or if the image has mode "1" or "P", it is set to be Inter.NEAREST.
            It can be any of [Inter.BILINEAR, Inter.NEAREST, Inter.BICUBIC].

            - Inter.BILINEAR, means resample method is bilinear interpolation.

            - Inter.NEAREST, means resample method is nearest-neighbor interpolation.

            - Inter.BICUBIC, means resample method is bicubic interpolation.

        fill_value (tuple or int, optional): Optional fill_value to fill the area outside the transform
            in the output image. Used only in Pillow versions > 5.0.0 (default=0, filling is performed).

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

    Examples:
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        >>>                          py_transforms.ToTensor()])
    """

    @check_random_affine
    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=Inter.NEAREST, fill_value=0):
        # Parameter checking
        # rotation
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        elif isinstance(degrees, (tuple, list)) and len(degrees) == 2:
            self.degrees = degrees
        else:
            raise TypeError("If degrees is a list or tuple, it must be of length 2.")

        # translation
        if translate is not None:
            if isinstance(translate, (tuple, list)) and len(translate) == 2:
                for t in translate:
                    if t < 0.0 or t > 1.0:
                        raise ValueError("translation values should be between 0 and 1")
            else:
                raise TypeError("translate should be a list or tuple of length 2.")
        self.translate = translate

        # scale
        if scale is not None:
            if isinstance(scale, (tuple, list)) and len(scale) == 2:
                for s in scale:
                    if s <= 0:
                        raise ValueError("scale values should be positive")
            else:
                raise TypeError("scale should be a list or tuple of length 2.")
        self.scale_ranges = scale

        # shear
        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-1 * shear, shear)
            elif isinstance(shear, (tuple, list)) and (len(shear) == 2 or len(shear) == 4):
                # X-Axis shear with [min, max]
                if len(shear) == 2:
                    self.shear = [shear[0], shear[1], 0., 0.]
                elif len(shear) == 4:
                    self.shear = [s for s in shear]
            else:
                raise TypeError("shear should be a list or tuple and it must be of length 2 or 4.")
        else:
            self.shear = shear

        # resample
        self.resample = DE_PY_INTER_MODE[resample]

        # fill_value
        self.fill_value = fill_value

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL Image): Image to be applied affine transformation.

        Returns:
            img (PIL Image), Randomly affine transformed image.
        """
        # rotation
        angle = random.uniform(self.degrees[0], self.degrees[1])

        # translation
        if self.translate is not None:
            max_dx = self.translate[0] * img.size[0]
            max_dy = self.translate[1] * img.size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        # scale
        if self.scale_ranges is not None:
            scale = random.uniform(self.scale_ranges[0], self.scale_ranges[1])
        else:
            scale = 1.0

        # shear
        if self.shear is not None:
            if len(self.shear) == 2:
                shear = [random.uniform(self.shear[0], self.shear[1]), 0.]
            elif len(self.shear) == 4:
                shear = [random.uniform(self.shear[0], self.shear[1]),
                         random.uniform(self.shear[2], self.shear[3])]
        else:
            shear = 0.0

        return util.random_affine(img,
                                  angle,
                                  translations,
                                  scale,
                                  shear,
                                  self.resample,
                                  self.fill_value)


class MixUp:
    """
    Apply mix up transformation to the input image and label, make one input data combined with others.

    Args:
        batch_size (int): the batch size of dataset.
        alpha (float):  the mix up rate.
        is_single (bool): for deciding using single batch or muti batch mix up transformation.
    """

    @check_mix_up
    def __init__(self, batch_size, alpha, is_single=True):
        self.image = 0
        self.label = 0
        self.is_first = True
        self.batch_size = batch_size
        self.alpha = alpha
        self.is_single = is_single

    def __call__(self, image, label):
        """
        Call method.

        Args:
            image (numpy.ndarray): numpy Image to be applied mix up transformation.
            label(numpy.ndarray): numpy label to be applied mix up transformation.

        Returns:
            image (numpy.ndarray): numpy Image after being applied mix up transformation.
            label(numpy.ndarray): numpy label after being applied mix up transformation.
        """
        if self.is_single:
            return util.mix_up_single(self.batch_size, image, label, self.alpha)
        return util.mix_up_muti(self, self.batch_size, image, label, self.alpha)


class RgbToHsv:
    """
    Convert a Numpy RGB image or one batch Numpy RGB images to HSV images.

    Args:
        is_hwc (bool): The flag of image shape, (H, W, C) or (N, H, W, C) if True
                       and (C, H, W) or (N, C, H, W) if False (default=False).
    """

    def __init__(self, is_hwc=False):
        self.is_hwc = is_hwc

    def __call__(self, rgb_imgs):
        """
        Call method.

        Args:
            rgb_imgs (numpy.ndarray): Numpy RGB images array of shape (H, W, C) or (N, H, W, C),
                                      or (C, H, W) or (N, C, H, W) to be converted.

        Returns:
            np_hsv_img (numpy.ndarray), Numpy HSV images with same shape of rgb_imgs.
        """
        return util.rgb_to_hsvs(rgb_imgs, self.is_hwc)


class HsvToRgb:
    """
    Convert a Numpy HSV image or one batch Numpy HSV images to RGB images.

    Args:
        is_hwc (bool): The flag of image shape, (H, W, C) or (N, H, W, C) if True
                       and (C, H, W) or (N, C, H, W) if False (default=False).
    """

    def __init__(self, is_hwc=False):
        self.is_hwc = is_hwc

    def __call__(self, hsv_imgs):
        """
        Call method.

        Args:
            hsv_imgs (numpy.ndarray): Numpy HSV images array of shape (H, W, C) or (N, H, W, C),
                                      or (C, H, W) or (N, C, H, W) to be converted.

        Returns:
            rgb_imgs (numpy.ndarray), Numpy RGB image with same shape of hsv_imgs.
        """
        return util.hsv_to_rgbs(hsv_imgs, self.is_hwc)


class RandomColor:
    """
    Adjust the color of the input PIL image by a random degree.

    Args:
        degrees (sequence): Range of random color adjustment degrees.
            It should be in (min, max) format (default=(0.1,1.9)).

    Examples:
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.RandomColor((0.5,1.5)),
        >>>                          py_transforms.ToTensor()])
    """

    def __init__(self, degrees=(0.1, 1.9)):
        self.degrees = degrees

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL Image): Image to be color adjusted.

        Returns:
            img (PIL Image), Color adjusted image.
        """

        return util.random_color(img, self.degrees)


class RandomSharpness:
    """
    Adjust the sharpness of the input PIL image by a random degree.

    Args:
        degrees (sequence): Range of random sharpness adjustment degrees.
            It should be in (min, max) format (default=(0.1,1.9)).

    Examples:
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.RandomSharpness((0.5,1.5)),
        >>>                          py_transforms.ToTensor()])

    """

    def __init__(self, degrees=(0.1, 1.9)):
        self.degrees = degrees

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL Image): Image to be sharpness adjusted.

        Returns:
            img (PIL Image), Color adjusted image.
        """

        return util.random_sharpness(img, self.degrees)


class AutoContrast:
    """
    Automatically maximize the contrast of the input PIL image.

    Examples:
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.AutoContrast(),
        >>>                          py_transforms.ToTensor()])

    """

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL Image): Image to be augmented with AutoContrast.

        Returns:
            img (PIL Image), Augmented image.
        """

        return util.auto_contrast(img)


class Invert:
    """
    Invert colors of input PIL image.

    Examples:
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.Invert(),
        >>>                          py_transforms.ToTensor()])

    """

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL Image): Image to be color Inverted.

        Returns:
            img (PIL Image), Color inverted image.
        """

        return util.invert_color(img)


class Equalize:
    """
    Equalize the histogram of input PIL image.

    Examples:
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.Equalize(),
        >>>                          py_transforms.ToTensor()])

    """

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL Image): Image to be equalized.

        Returns:
            img (PIL Image), Equalized image.
        """

        return util.equalize(img)


class UniformAugment:
    """
    Uniformly select and apply a number of transforms sequentially from
    a list of transforms. Randomly assigns a probability to each transform for
    each image to decide whether apply it or not.

    Args:
         transforms (list): List of transformations to be chosen from to apply.
         num_ops (int, optional): number of transforms to sequentially apply (default=2).

    Examples:
        >>> transforms_list = [py_transforms.CenterCrop(64),
        >>>                    py_transforms.RandomColor(),
        >>>                    py_transforms.RandomSharpness(),
        >>>                    py_transforms.RandomRotation(30)]
        >>> py_transforms.ComposeOp([py_transforms.Decode(),
        >>>                          py_transforms.UniformAugment(transforms_list),
        >>>                          py_transforms.ToTensor()])
    """

    def __init__(self, transforms, num_ops=2):
        self.transforms = transforms
        self.num_ops = num_ops

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL Image): Image to be applied transformation.

        Returns:
            img (PIL Image), Transformed image.
        """
        return util.uniform_augment(img, self.transforms.copy(), self.num_ops)
