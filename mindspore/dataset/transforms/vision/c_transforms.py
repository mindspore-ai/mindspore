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
The module vision.c_transforms is inheritted from _c_dataengine
which is implemented basing on opencv in C++. It's a high performance module to
process image augmentations. Users can apply suitable augmentations on image data
to improve their training models.

.. Note::
    Constructor's arguments for every class in this module must be saved into the
    class attributes (self.xxx) to support save() and load().

Examples:
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.transforms.c_transforms as c_transforms
        >>> import mindspore.dataset.transforms.vision.c_transforms as vision
        >>> dataset_dir = "path/to/imagefolder_directory"
        >>> # create a dataset that reads all files in dataset_dir with 8 threads
        >>> dataset = ds.ImageFolderDatasetV2(dataset_dir, num_parallel_workers=8)
        >>> # create a list of transformations to be applied to the image data
        >>> transforms_list = [vision.Decode(),
        >>>                    vision.Resize((256, 256)),
        >>>                    vision.RandomRotation((0, 15)),
        >>>                    vision.Normalize((100,  115.0, 121.0), (71.0, 68.0, 70.0)),
        >>>                    vision.HWC2CHW()]
        >>> onehot_op = c_transforms.OneHot(num_classes)
        >>> # apply the transform to the dataset through dataset.map()
        >>> dataset = dataset.map(input_columns="image", operations=transforms_list)
        >>> dataset = dataset.map(input_columns="label", operations=onehot_op)
"""
import mindspore._c_dataengine as cde

from .utils import Inter, Border
from .validators import check_prob, check_crop, check_resize_interpolation, check_random_resize_crop, \
    check_normalize_c, check_random_crop, check_random_color_adjust, check_random_rotation, \
    check_resize, check_rescale, check_pad, check_cutout

DE_C_INTER_MODE = {Inter.NEAREST: cde.InterpolationMode.DE_INTER_NEAREST_NEIGHBOUR,
                   Inter.LINEAR: cde.InterpolationMode.DE_INTER_LINEAR,
                   Inter.CUBIC: cde.InterpolationMode.DE_INTER_CUBIC}

DE_C_BORDER_TYPE = {Border.CONSTANT: cde.BorderType.DE_BORDER_CONSTANT,
                    Border.EDGE: cde.BorderType.DE_BORDER_EDGE,
                    Border.REFLECT: cde.BorderType.DE_BORDER_REFLECT,
                    Border.SYMMETRIC: cde.BorderType.DE_BORDER_SYMMETRIC}


class Decode(cde.DecodeOp):
    """
    Decode the input image in RGB mode.
    """

    def __init__(self, rgb=True):
        self.rgb = rgb
        super().__init__(self.rgb)


class CutOut(cde.CutOutOp):
    """
    Randomly cut (mask) out a given number of square patches from the input Numpy image array.

    Args:
        length (int): The side length of each square patch.
        num_patches (int, optional): Number of patches to be cut out of an image (default=1).
    """

    @check_cutout
    def __init__(self, length, num_patches=1):
        self.length = length
        self.num_patches = num_patches
        fill_value = (0, 0, 0)
        super().__init__(length, length, num_patches, False, *fill_value)


class Normalize(cde.NormalizeOp):
    """
    Normalize the input image with respect to mean and standard deviation.

    Args:
        mean (list): List of mean values for each channel, w.r.t channel order.
        std (list): List of standard deviations for each channel, w.r.t. channel order.
    """

    @check_normalize_c
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        super().__init__(*mean, *std)


class RandomCrop(cde.RandomCropOp):
    """
    Crop the input image at a random location.

    Args:
        size (int or sequence): The output size of the cropped image.
            If size is an int, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).
        padding (int or sequence, optional): The number of pixels to pad the image (default=None).
            If a single number is provided, it pads all borders with this value.
            If a tuple or list of 2 values are provided, it pads the (left and top)
            with the first value and (right and bottom) with the second value.
            If 4 values are provided as a list or tuple,
            it pads the left, top, right and bottom respectively.
        pad_if_needed (bool, optional): Pad the image if either side is smaller than
            the given output size (default=False).
        fill_value (int or tuple, optional): The pixel intensity of the borders if
            the padding_mode is Border.CONSTANT (default=0). If it is a 3-tuple, it is used to
            fill R, G, B channels respectively.
        padding_mode (Border mode, optional): The method of padding (default=Border.CONSTANT). Can be any of
            [Border.CONSTANT, Border.EDGE, Border.REFLECT, Border.SYMMETRIC].

            - Border.CONSTANT, means it fills the border with constant values.

            - Border.EDGE, means it pads with the last value on the edge.

            - Border.REFLECT, means it reflects the values on the edge omitting the last
              value of edge.

            - Border.SYMMETRIC, means it reflects the values on the edge repeating the last
              value of edge.
    """

    @check_random_crop
    def __init__(self, size, padding=None, pad_if_needed=False, fill_value=0, padding_mode=Border.CONSTANT):
        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill_value = fill_value
        self.padding_mode = padding_mode.value
        if padding is None:
            padding = (0, 0, 0, 0)
        if isinstance(fill_value, int):  # temporary fix
            fill_value = tuple([fill_value] * 3)
        border_type = DE_C_BORDER_TYPE[padding_mode]
        super().__init__(*size, *padding, border_type, pad_if_needed, *fill_value)


class RandomHorizontalFlip(cde.RandomHorizontalFlipOp):
    """
    Flip the input image horizontally, randomly with a given probability.

    Args:
        prob (float): Probability of the image being flipped (default=0.5).
    """

    @check_prob
    def __init__(self, prob=0.5):
        self.prob = prob
        super().__init__(prob)


class RandomVerticalFlip(cde.RandomVerticalFlipOp):
    """
    Flip the input image vertically, randomly with a given probability.

    Args:
        prob (float): Probability of the image being flipped (default=0.5).
    """

    @check_prob
    def __init__(self, prob=0.5):
        self.prob = prob
        super().__init__(prob)


class Resize(cde.ResizeOp):
    """
    Resize the input image to the given size.

    Args:
        size (int or sequence): The output size of the resized image.
            If size is an int, smaller edge of the image will be resized to this value with
            the same image aspect ratio.
            If size is a sequence of length 2, it should be (height, width).
        interpolation (Inter mode, optional): Image interpolation mode (default=Inter.LINEAR).
            It can be any of [Inter.LINEAR, Inter.NEAREST, Inter.BICUBIC].

            - Inter.LINEAR, means interpolation method is bilinear interpolation.

            - Inter.NEAREST, means interpolation method is nearest-neighbor interpolation.

            - Inter.BICUBIC, means interpolation method is bicubic interpolation.
    """

    @check_resize_interpolation
    def __init__(self, size, interpolation=Inter.LINEAR):
        self.size = size
        self.interpolation = interpolation
        interpoltn = DE_C_INTER_MODE[interpolation]
        if isinstance(size, int):
            super().__init__(size, interpolation=interpoltn)
        else:
            super().__init__(*size, interpoltn)


class RandomResizedCrop(cde.RandomCropAndResizeOp):
    """
    Crop the input image to a random size and aspect ratio.

    Args:
        size (int or sequence): The size of the output image.
            If size is an int, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).
        scale (tuple, optional): Range (min, max) of respective size of the original
            size to be cropped (default=(0.08, 1.0)).
        ratio (tuple, optional): Range (min, max) of aspect ratio to be cropped
            (default=(3. / 4., 4. / 3.)).
        interpolation (Inter mode, optional): Image interpolation mode (default=Inter.BILINEAR).
            It can be any of [Inter.BILINEAR, Inter.NEAREST, Inter.BICUBIC].

            - Inter.BILINEAR, means interpolation method is bilinear interpolation.

            - Inter.NEAREST, means interpolation method is nearest-neighbor interpolation.

            - Inter.BICUBIC, means interpolation method is bicubic interpolation.

        max_attempts (int, optional): The maximum number of attempts to propose a valid
            crop_area (default=10). If exceeded, fall back to use center_crop instead.
    """

    @check_random_resize_crop
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation=Inter.BILINEAR, max_attempts=10):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.max_attempts = max_attempts
        interpoltn = DE_C_INTER_MODE[interpolation]
        super().__init__(*size, *scale, *ratio, interpoltn, max_attempts)


class CenterCrop(cde.CenterCropOp):
    """
    Crops the input image at the center to the given size.

    Args:
        size (int or sequence): The output size of the cropped image.
            If size is an int, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).
    """

    @check_crop
    def __init__(self, size):
        self.size = size
        super().__init__(*size)


class RandomColorAdjust(cde.RandomColorAdjustOp):
    """
    Randomly adjust the brightness, contrast, saturation, and hue of the input image.

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
    """

    @check_random_color_adjust
    def __init__(self, brightness=(1, 1), contrast=(1, 1), saturation=(1, 1), hue=(0, 0)):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        super().__init__(*brightness, *contrast, *saturation, *hue)


class RandomRotation(cde.RandomRotationOp):
    """
    Rotate the input image by a random angle.

    Args:
        degrees (int or float or sequence): Range of random rotation degrees.
            If degrees is a number, the range will be converted to (-degrees, degrees).
            If degrees is a sequence, it should be (min, max).
        resample (Inter mode, optional): An optional resampling filter (default=Inter.NEAREST).
            If omitted, or if the image has mode "1" or "P", it is set to be Inter.NEAREST.
            It can be any of [Inter.BILINEAR, Inter.NEAREST, Inter.BICUBIC].

            - Inter.BILINEAR, means resample method is bilinear interpolation.

            - Inter.NEAREST, means resample method is nearest-neighbor interpolation.

            - Inter.BICUBIC, means resample method is bicubic interpolation.

        expand (bool, optional):  Optional expansion flag (default=False). If set to True, expand the output
            image to make it large enough to hold the entire rotated image.
            If set to False or omitted, make the output image the same size as the input.
            Note that the expand flag assumes rotation around the center and no translation.
        center (tuple, optional): Optional center of rotation (a 2-tuple) (default=None).
            Origin is the top left corner. None sets to the center of the image.
        fill_value (int or tuple, optional): Optional fill color for the area outside the rotated image (default=0).
            If it is a 3-tuple, it is used for R, G, B channels respectively.
            If it is an int, it is used for all RGB channels.
    """

    @check_random_rotation
    def __init__(self, degrees, resample=Inter.NEAREST, expand=False, center=None, fill_value=0):
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill_value = fill_value
        if center is None:
            center = (-1, -1)
        if isinstance(fill_value, int):  # temporary fix
            fill_value = tuple([fill_value] * 3)
        interpolation = DE_C_INTER_MODE[resample]
        super().__init__(*degrees, *center, interpolation, expand, *fill_value)


class Rescale(cde.RescaleOp):
    """
    Tensor operation to rescale the input image.

    Args:
        rescale (float): Rescale factor.
        shift (float): Shift factor.
    """

    @check_rescale
    def __init__(self, rescale, shift):
        self.rescale = rescale
        self.shift = shift
        super().__init__(rescale, shift)


class RandomResize(cde.RandomResizeOp):
    """
    Tensor operation to resize the input image using a randomly selected interpolation mode.

    Args:
        size (int or sequence): The output size of the resized image.
            If size is an int, smaller edge of the image will be resized to this value with
            the same image aspect ratio.
            If size is a sequence of length 2, it should be (height, width).
    """

    @check_resize
    def __init__(self, size):
        self.size = size
        if isinstance(size, int):
            super().__init__(size)
        else:
            super().__init__(*size)


class HWC2CHW(cde.ChannelSwapOp):
    """
    Transpose the input image; shape (H, W, C) to shape (C, H, W).
    """


class RandomCropDecodeResize(cde.RandomCropDecodeResizeOp):
    """
    Equivalent to RandomResizedCrop, but crops before decodes.

    Args:
        size (int or sequence, optional): The size of the output image.
            If size is an int, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).
        scale (tuple, optional): Range (min, max) of respective size of the
            original size to be cropped (default=(0.08, 1.0)).
        ratio (tuple, optional): Range (min, max) of aspect ratio to be
            cropped (default=(3. / 4., 4. / 3.)).
        interpolation (Inter mode, optional): Image interpolation mode (default=Inter.BILINEAR).
            It can be any of [Inter.BILINEAR, Inter.NEAREST, Inter.BICUBIC].

            - Inter.BILINEAR, means interpolation method is bilinear interpolation.

            - Inter.NEAREST, means interpolation method is nearest-neighbor interpolation.

            - Inter.BICUBIC, means interpolation method is bicubic interpolation.

        max_attempts (int, optional): The maximum number of attempts to propose a valid crop_area (default=10).
            If exceeded, fall back to use center_crop instead.
    """

    @check_random_resize_crop
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation=Inter.BILINEAR, max_attempts=10):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.max_attempts = max_attempts
        interpoltn = DE_C_INTER_MODE[interpolation]
        super().__init__(*size, *scale, *ratio, interpoltn, max_attempts)


class Pad(cde.PadOp):
    """
    Pads the image according to padding parameters.

    Args:
        padding (int or sequence): The number of pixels to pad the image.
            If a single number is provided, it pads all borders with this value.
            If a tuple or list of 2 values are provided, it pads the (left and top)
            with the first value and (right and bottom) with the second value.
            If 4 values are provided as a list or tuple,
            it pads the left, top, right and bottom respectively.
        fill_value (int or tuple, optional): The pixel intensity of the borders if
            the padding_mode is Border.CONSTANT (default=0). If it is a 3-tuple, it is used to
            fill R, G, B channels respectively.
        padding_mode (Border mode): The method of padding (default=Border.CONSTANT). Can be any of
            [Border.CONSTANT, Border.EDGE, Border.REFLECT, Border.SYMMETRIC].

            - Border.CONSTANT, means it fills the border with constant values.

            - Border.EDGE, means it pads with the last value on the edge.

            - Border.REFLECT, means it reflects the values on the edge omitting the last
              value of edge.

            - Border.SYMMETRIC, means it reflects the values on the edge repeating the last
              value of edge.
    """

    @check_pad
    def __init__(self, padding, fill_value=0, padding_mode=Border.CONSTANT):
        self.padding = padding
        self.fill_value = fill_value
        self.padding_mode = padding_mode
        if isinstance(fill_value, int):  # temporary fix
            fill_value = tuple([fill_value] * 3)
        padding_mode = DE_C_BORDER_TYPE[padding_mode]
        super().__init__(*padding, padding_mode, *fill_value)
