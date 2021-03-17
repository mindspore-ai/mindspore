# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
The module vision.c_transforms is inherited from _c_dataengine
and is implemented based on OpenCV in C++. It's a high performance module to
process images. Users can apply suitable augmentations on image data
to improve their training models.

.. Note::
    A constructor's arguments for every class in this module must be saved into the
    class attributes (self.xxx) to support save() and load().

Examples:
    >>> from mindspore.dataset.vision import Border, Inter
    >>> image_folder_dataset_dir = "/path/to/image_folder_dataset_directory"
    >>> # create a dataset that reads all files in dataset_dir with 8 threads
    >>> image_folder_dataset = ds.ImageFolderDataset(image_folder_dataset_dir,
    ...                                              num_parallel_workers=8)
    >>> # create a list of transformations to be applied to the image data
    >>> transforms_list = [c_vision.Decode(),
    ...                    c_vision.Resize((256, 256), interpolation=Inter.LINEAR),
    ...                    c_vision.RandomCrop(200, padding_mode=Border.EDGE),
    ...                    c_vision.RandomRotation((0, 15)),
    ...                    c_vision.Normalize((100, 115.0, 121.0), (71.0, 68.0, 70.0)),
    ...                    c_vision.HWC2CHW()]
    >>> onehot_op = c_transforms.OneHot(num_classes=10)
    >>> # apply the transformation to the dataset through data1.map()
    >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
    ...                                                 input_columns="image")
    >>> image_folder_dataset = image_folder_dataset.map(operations=onehot_op,
    ...                                                 input_columns="label")
"""
import numbers
import numpy as np
from PIL import Image
import mindspore._c_dataengine as cde

from .utils import Inter, Border, ImageBatchFormat
from .validators import check_prob, check_crop, check_resize_interpolation, check_random_resize_crop, \
    check_mix_up_batch_c, check_normalize_c, check_normalizepad_c, check_random_crop, check_random_color_adjust, \
    check_random_rotation, check_range, check_resize, check_rescale, check_pad, check_cutout, \
    check_uniform_augment_cpp, \
    check_bounding_box_augment_cpp, check_random_select_subpolicy_op, check_auto_contrast, check_random_affine, \
    check_random_solarize, check_soft_dvpp_decode_random_crop_resize_jpeg, check_positive_degrees, FLOAT_MAX_INTEGER, \
    check_cut_mix_batch_c, check_posterize
from ..transforms.c_transforms import TensorOperation


class ImageTensorOperation(TensorOperation):
    """
    Base class of Image Tensor Ops
    """

    def __call__(self, *input_tensor_list):
        for tensor in input_tensor_list:
            if not isinstance(tensor, (np.ndarray, Image.Image)):
                raise TypeError("Input should be NumPy or PIL image, got {}.".format(type(tensor)))
        return super().__call__(*input_tensor_list)

    def parse(self):
        raise NotImplementedError("ImageTensorOperation has to implement parse() method.")


DE_C_BORDER_TYPE = {Border.CONSTANT: cde.BorderType.DE_BORDER_CONSTANT,
                    Border.EDGE: cde.BorderType.DE_BORDER_EDGE,
                    Border.REFLECT: cde.BorderType.DE_BORDER_REFLECT,
                    Border.SYMMETRIC: cde.BorderType.DE_BORDER_SYMMETRIC}

DE_C_IMAGE_BATCH_FORMAT = {ImageBatchFormat.NHWC: cde.ImageBatchFormat.DE_IMAGE_BATCH_FORMAT_NHWC,
                           ImageBatchFormat.NCHW: cde.ImageBatchFormat.DE_IMAGE_BATCH_FORMAT_NCHW}

DE_C_INTER_MODE = {Inter.NEAREST: cde.InterpolationMode.DE_INTER_NEAREST_NEIGHBOUR,
                   Inter.LINEAR: cde.InterpolationMode.DE_INTER_LINEAR,
                   Inter.CUBIC: cde.InterpolationMode.DE_INTER_CUBIC,
                   Inter.AREA: cde.InterpolationMode.DE_INTER_AREA}


def parse_padding(padding):
    if isinstance(padding, numbers.Number):
        padding = [padding] * 4
    if len(padding) == 2:
        left = top = padding[0]
        right = bottom = padding[1]
        padding = (left, top, right, bottom,)
    if isinstance(padding, list):
        padding = tuple(padding)
    return padding


class AutoContrast(ImageTensorOperation):
    """
    Apply automatic contrast on input image.

    Args:
        cutoff (float, optional): Percent of pixels to cut off from the histogram,
            the value must be in the range [0.0, 50.0) (default=0.0).
        ignore (Union[int, sequence], optional): Pixel values to ignore (default=None).

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.AutoContrast(cutoff=10.0, ignore=[10, 20])]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_auto_contrast
    def __init__(self, cutoff=0.0, ignore=None):
        if ignore is None:
            ignore = []
        if isinstance(ignore, int):
            ignore = [ignore]
        self.cutoff = cutoff
        self.ignore = ignore

    def parse(self):
        return cde.AutoContrastOperation(self.cutoff, self.ignore)


class BoundingBoxAugment(ImageTensorOperation):
    """
    Apply a given image transform on a random selection of bounding box regions of a given image.

    Args:
        transform: C++ transformation function to be applied on random selection
            of bounding box regions of a given image.
        ratio (float, optional): Ratio of bounding boxes to apply augmentation on.
            Range: [0, 1] (default=0.3).

    Examples:
        >>> # set bounding box operation with ratio of 1 to apply rotation on all bounding boxes
        >>> bbox_aug_op = c_vision.BoundingBoxAugment(c_vision.RandomRotation(90), 1)
        >>> # map to apply ops
        >>> image_folder_dataset = image_folder_dataset.map(operations=[bbox_aug_op],
        ...                                                 input_columns=["image", "bbox"],
        ...                                                 output_columns=["image", "bbox"],
        ...                                                 column_order=["image", "bbox"])
    """

    @check_bounding_box_augment_cpp
    def __init__(self, transform, ratio=0.3):
        self.ratio = ratio
        self.transform = transform

    def parse(self):
        if self.transform and getattr(self.transform, 'parse', None):
            transform = self.transform.parse()
        else:
            transform = self.transform
        return cde.BoundingBoxAugmentOperation(transform, self.ratio)


class CenterCrop(ImageTensorOperation):
    """
    Crop the input image at the center to the given size.

    Args:
        size (Union[int, sequence]): The output size of the cropped image.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).

    Examples:
        >>> # crop image to a square
        >>> transforms_list1 = [c_vision.Decode(), c_vision.CenterCrop(50)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list1,
        ...                                                 input_columns=["image"])
        >>> # crop image to portrait style
        >>> transforms_list2 = [c_vision.Decode(), c_vision.CenterCrop((60, 40))]
        >>> image_folder_dataset_1 = image_folder_dataset_1.map(operations=transforms_list2,
        ...                                                     input_columns=["image"])
    """

    @check_crop
    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def parse(self):
        return cde.CenterCropOperation(self.size)


class CutMixBatch(ImageTensorOperation):
    """
    Apply CutMix transformation on input batch of images and labels.
    Note that you need to make labels into one-hot format and batch before calling this function.

    Args:
        image_batch_format (Image Batch Format): The method of padding. Can be any of
            [ImageBatchFormat.NHWC, ImageBatchFormat.NCHW]
        alpha (float, optional): hyperparameter of beta distribution (default = 1.0).
        prob (float, optional): The probability by which CutMix is applied to each image (default = 1.0).

    Examples:
        >>> from mindspore.dataset.vision import ImageBatchFormat
        >>> onehot_op = c_transforms.OneHot(num_classes=10)
        >>> image_folder_dataset= image_folder_dataset.map(operations=onehot_op,
        ...                                                input_columns=["label"])
        >>> cutmix_batch_op = c_vision.CutMixBatch(ImageBatchFormat.NHWC, 1.0, 0.5)
        >>> image_folder_dataset = image_folder_dataset.batch(5)
        >>> image_folder_dataset = image_folder_dataset.map(operations=cutmix_batch_op,
        ...                                                 input_columns=["image", "label"])
    """

    @check_cut_mix_batch_c
    def __init__(self, image_batch_format, alpha=1.0, prob=1.0):
        self.image_batch_format = image_batch_format.value
        self.alpha = alpha
        self.prob = prob

    def parse(self):
        return cde.CutMixBatchOperation(DE_C_IMAGE_BATCH_FORMAT[self.image_batch_format], self.alpha, self.prob)


class CutOut(ImageTensorOperation):
    """
    Randomly cut (mask) out a given number of square patches from the input NumPy image array.

    Args:
        length (int): The side length of each square patch.
        num_patches (int, optional): Number of patches to be cut out of an image (default=1).

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.CutOut(80, num_patches=10)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_cutout
    def __init__(self, length, num_patches=1):
        self.length = length
        self.num_patches = num_patches

    def parse(self):
        return cde.CutOutOperation(self.length, self.num_patches)


class Decode(ImageTensorOperation):
    """
    Decode the input image in RGB mode.

    Args:
        rgb (bool, optional): Mode of decoding input image (default=True).
            If True means format of decoded image is RGB else BGR(deprecated).

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomHorizontalFlip()]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    def __init__(self, rgb=True):
        self.rgb = rgb

    def __call__(self, img):
        """
        Call method.

        Args:
            img (NumPy): Image to be decoded.

        Returns:
            img (NumPy), Decoded image.
        """
        if not isinstance(img, np.ndarray) or img.ndim != 1 or img.dtype.type is np.str_:
            raise TypeError("Input should be an encoded image in 1-D NumPy format, got {}.".format(type(img)))
        return super().__call__(img)

    def parse(self):
        return cde.DecodeOperation(self.rgb)


class Equalize(ImageTensorOperation):
    """
    Apply histogram equalization on input image.

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.Equalize()]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    def parse(self):
        return cde.EqualizeOperation()


class HWC2CHW(ImageTensorOperation):
    """
    Transpose the input image; shape (H, W, C) to shape (C, H, W).

    Examples:
        >>> transforms_list = [c_vision.Decode(),
        ...                    c_vision.RandomHorizontalFlip(0.75),
        ...                    c_vision.RandomCrop(512),
        ...                    c_vision.HWC2CHW()]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    def parse(self):
        return cde.HwcToChwOperation()


class Invert(ImageTensorOperation):
    """
    Apply invert on input image in RGB mode.

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.Invert()]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    def parse(self):
        return cde.InvertOperation()


class MixUpBatch(ImageTensorOperation):
    """
    Apply MixUp transformation on input batch of images and labels. Each image is multiplied by a random weight (lambda)
    and then added to a randomly selected image from the batch multiplied by (1 - lambda). The same formula is also
    applied to the one-hot labels.
    Note that you need to make labels into one-hot format and batch before calling this function.

    Args:
        alpha (float, optional): Hyperparameter of beta distribution (default = 1.0).

    Examples:
        >>> onehot_op = c_transforms.OneHot(num_classes=10)
        >>> image_folder_dataset= image_folder_dataset.map(operations=onehot_op,
        ...                                                input_columns=["label"])
        >>> mixup_batch_op = c_vision.MixUpBatch(alpha=0.9)
        >>> image_folder_dataset = image_folder_dataset.batch(5)
        >>> image_folder_dataset = image_folder_dataset.map(operations=mixup_batch_op,
        ...                                                 input_columns=["image", "label"])
    """

    @check_mix_up_batch_c
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def parse(self):
        return cde.MixUpBatchOperation(self.alpha)


class Normalize(ImageTensorOperation):
    """
    Normalize the input image with respect to mean and standard deviation.

    Args:
        mean (sequence): List or tuple of mean values for each channel, with respect to channel order.
            The mean values must be in range [0.0, 255.0].
        std (sequence): List or tuple of standard deviations for each channel, with respect to channel order.
            The standard deviation values must be in range (0.0, 255.0].

    Examples:
        >>> decode_op = c_vision.Decode()
        >>> normalize_op = c_vision.Normalize(mean=[121.0, 115.0, 100.0], std=[70.0, 68.0, 71.0])
        >>> transforms_list = [decode_op, normalize_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_normalize_c
    def __init__(self, mean, std):
        if len(mean) == 1:
            mean = [mean[0]] * 3
        if len(std) == 1:
            std = [std[0]] * 3
        self.mean = mean
        self.std = std

    def parse(self):
        return cde.NormalizeOperation(self.mean, self.std)


class NormalizePad(ImageTensorOperation):
    """
    Normalize the input image with respect to mean and standard deviation then pad an extra channel with value zero.

    Args:
        mean (sequence): List or tuple of mean values for each channel, with respect to channel order.
            The mean values must be in range (0.0, 255.0].
        std (sequence): List or tuple of standard deviations for each channel, with respect to channel order.
            The standard deviation values must be in range (0.0, 255.0].
        dtype (str): Set the output data type of normalized image (default is "float32").

    Examples:
        >>> decode_op = c_vision.Decode()
        >>> normalize_pad_op = c_vision.NormalizePad(mean=[121.0, 115.0, 100.0],
        ...                                          std=[70.0, 68.0, 71.0],
        ...                                          dtype="float32")
        >>> transforms_list = [decode_op, normalize_pad_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_normalizepad_c
    def __init__(self, mean, std, dtype="float32"):
        self.mean = mean
        self.std = std
        self.dtype = dtype

    def parse(self):
        return cde.NormalizePadOperation(self.mean, self.std, self.dtype)


class Pad(ImageTensorOperation):
    """
    Pad the image according to padding parameters.

    Args:
        padding (Union[int, sequence]): The number of pixels to pad the image.
            If a single number is provided, it pads all borders with this value.
            If a tuple or list of 2 values are provided, it pads the (left and top)
            with the first value and (right and bottom) with the second value.
            If 4 values are provided as a list or tuple,
            it pads the left, top, right and bottom respectively.
        fill_value (Union[int, tuple], optional): The pixel intensity of the borders, only valid for
            padding_mode Border.CONSTANT. If it is a 3-tuple, it is used to fill R, G, B channels respectively.
            If it is an integer, it is used for all RGB channels.
            The fill_value values must be in range [0, 255] (default=0).
        padding_mode (Border mode, optional): The method of padding (default=Border.CONSTANT). Can be any of
            [Border.CONSTANT, Border.EDGE, Border.REFLECT, Border.SYMMETRIC].

            - Border.CONSTANT, means it fills the border with constant values.

            - Border.EDGE, means it pads with the last value on the edge.

            - Border.REFLECT, means it reflects the values on the edge omitting the last
              value of edge.

            - Border.SYMMETRIC, means it reflects the values on the edge repeating the last
              value of edge.

    Examples:
        >>> from mindspore.dataset.vision import Border
        >>> transforms_list = [c_vision.Decode(), c_vision.Pad([100, 100, 100, 100])]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_pad
    def __init__(self, padding, fill_value=0, padding_mode=Border.CONSTANT):
        padding = parse_padding(padding)
        if isinstance(fill_value, int):
            fill_value = tuple([fill_value] * 3)
        self.padding = padding
        self.fill_value = fill_value
        self.padding_mode = padding_mode

    def parse(self):
        return cde.PadOperation(self.padding, self.fill_value, DE_C_BORDER_TYPE[self.padding_mode])


class RandomAffine(ImageTensorOperation):
    """
    Apply Random affine transformation to the input image.

    Args:
        degrees (int or float or sequence): Range of the rotation degrees.
            If degrees is a number, the range will be (-degrees, degrees).
            If degrees is a sequence, it should be (min, max).
        translate (sequence, optional): Sequence (tx_min, tx_max, ty_min, ty_max) of minimum/maximum translation in
            x(horizontal) and y(vertical) directions (default=None).
            The horizontal and vertical shift is selected randomly from the range:
            (tx_min*width, tx_max*width) and (ty_min*height, ty_max*height), respectively.
            If a tuple or list of size 2, then a translate parallel to the X axis in the range of
            (translate[0], translate[1]) is applied.
            If a tuple of list of size 4, then a translate parallel to the X axis in the range of
            (translate[0], translate[1]) and a translate parallel to the Y axis in the range of
            (translate[2], translate[3]) are applied.
            If None, no translation is applied.
        scale (sequence, optional): Scaling factor interval (default=None, original scale is used).
        shear (int or float or sequence, optional): Range of shear factor (default=None).
            If a number, then a shear parallel to the X axis in the range of (-shear, +shear) is applied.
            If a tuple or list of size 2, then a shear parallel to the X axis in the range of (shear[0], shear[1])
            is applied.
            If a tuple of list of size 4, then a shear parallel to X axis in the range of (shear[0], shear[1])
            and a shear parallel to Y axis in the range of (shear[2], shear[3]) is applied.
            If None, no shear is applied.
        resample (Inter mode, optional): An optional resampling filter (default=Inter.NEAREST).
            If omitted, or if the image has mode "1" or "P", it is set to be Inter.NEAREST.
            It can be any of [Inter.BILINEAR, Inter.NEAREST, Inter.BICUBIC].

            - Inter.BILINEAR, means resample method is bilinear interpolation.

            - Inter.NEAREST, means resample method is nearest-neighbor interpolation.

            - Inter.BICUBIC, means resample method is bicubic interpolation.

        fill_value (tuple or int, optional): Optional fill_value to fill the area outside the transform
            in the output image. There must be three elements in tuple and the value of single element is [0, 255].
            Used only in Pillow versions > 5.0.0 (default=0, filling is performed).

    Raises:
        ValueError: If degrees is negative.
        ValueError: If translation value is not between -1 and 1.
        ValueError: If scale is not positive.
        ValueError: If shear is a number but is not positive.
        TypeError: If degrees is not a number or a list or a tuple.
            If degrees is a list or tuple, its length is not 2.
        TypeError: If translate is specified but is not list or a tuple of length 2 or 4.
        TypeError: If scale is not a list or tuple of length 2.''
        TypeError: If shear is not a list or tuple of length 2 or 4.
        TypeError: If fill_value is not a single integer or a 3-tuple.

    Examples:
        >>> from mindspore.dataset.vision import Inter
        >>> decode_op = c_vision.Decode()
        >>> random_affine_op = c_vision.RandomAffine(degrees=15,
        ...                                          translate=(-0.1, 0.1, 0, 0),
        ...                                          scale=(0.9, 1.1),
        ...                                          resample=Inter.NEAREST)
        >>> transforms_list = [decode_op, random_affine_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_random_affine
    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=Inter.NEAREST, fill_value=0):
        # Parameter checking
        if shear is not None:
            if isinstance(shear, numbers.Number):
                shear = (-1 * shear, shear, 0., 0.)
            else:
                if len(shear) == 2:
                    shear = [shear[0], shear[1], 0., 0.]
                elif len(shear) == 4:
                    shear = [s for s in shear]

        if isinstance(degrees, numbers.Number):
            degrees = (-1 * degrees, degrees)

        if isinstance(fill_value, numbers.Number):
            fill_value = (fill_value, fill_value, fill_value)

        # translation
        if translate is None:
            translate = (0.0, 0.0, 0.0, 0.0)

        # scale
        if scale is None:
            scale = (1.0, 1.0)

        # shear
        if shear is None:
            shear = (0.0, 0.0, 0.0, 0.0)

        self.degrees = degrees
        self.translate = translate
        self.scale_ = scale
        self.shear = shear
        self.resample = DE_C_INTER_MODE[resample]
        self.fill_value = fill_value

    def parse(self):
        return cde.RandomAffineOperation(self.degrees, self.translate, self.scale_, self.shear, self.resample,
                                         self.fill_value)


class RandomColor(ImageTensorOperation):
    """
    Adjust the color of the input image by a fixed or random degree.
    This operation works only with 3-channel color images.

    Args:
         degrees (sequence, optional): Range of random color adjustment degrees.
            It should be in (min, max) format. If min=max, then it is a
            single fixed magnitude operation (default=(0.1, 1.9)).

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomColor((0.5, 2.0))]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_positive_degrees
    def __init__(self, degrees=(0.1, 1.9)):
        self.degrees = degrees

    def parse(self):
        return cde.RandomColorOperation(*self.degrees)


class RandomColorAdjust(ImageTensorOperation):
    """
    Randomly adjust the brightness, contrast, saturation, and hue of the input image.

    Args:
        brightness (Union[float, list, tuple], optional): Brightness adjustment factor (default=(1, 1)).
            Cannot be negative.
            If it is a float, the factor is uniformly chosen from the range [max(0, 1-brightness), 1+brightness].
            If it is a sequence, it should be [min, max] for the range.
        contrast (Union[float, list, tuple], optional): Contrast adjustment factor (default=(1, 1)).
            Cannot be negative.
            If it is a float, the factor is uniformly chosen from the range [max(0, 1-contrast), 1+contrast].
            If it is a sequence, it should be [min, max] for the range.
        saturation (Union[float, list, tuple], optional): Saturation adjustment factor (default=(1, 1)).
            Cannot be negative.
            If it is a float, the factor is uniformly chosen from the range [max(0, 1-saturation), 1+saturation].
            If it is a sequence, it should be [min, max] for the range.
        hue (Union[float, list, tuple], optional): Hue adjustment factor (default=(0, 0)).
            If it is a float, the range will be [-hue, hue]. Value should be 0 <= hue <= 0.5.
            If it is a sequence, it should be [min, max] where -0.5 <= min <= max <= 0.5.

    Examples:
        >>> decode_op = c_vision.Decode()
        >>> transform_op = c_vision.RandomColorAdjust(brightness=(0.5, 1),
        ...                                           contrast=(0.4, 1),
        ...                                           saturation=(0.3, 1))
        >>> transforms_list = [decode_op, transform_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_random_color_adjust
    def __init__(self, brightness=(1, 1), contrast=(1, 1), saturation=(1, 1), hue=(0, 0)):
        brightness = self.expand_values(brightness)
        contrast = self.expand_values(contrast)
        saturation = self.expand_values(saturation)
        hue = self.expand_values(hue, center=0, bound=(-0.5, 0.5), non_negative=False)

        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def expand_values(self, value, center=1, bound=(0, FLOAT_MAX_INTEGER), non_negative=True):
        if isinstance(value, numbers.Number):
            value = [center - value, center + value]
            if non_negative:
                value[0] = max(0, value[0])
            check_range(value, bound)
        return (value[0], value[1])

    def parse(self):
        return cde.RandomColorAdjustOperation(self.brightness, self.contrast, self.saturation, self.hue)


class RandomCrop(ImageTensorOperation):
    """
    Crop the input image at a random location.


    Args:
        size (Union[int, sequence]): The output size of the cropped image.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).
        padding (Union[int, sequence], optional): The number of pixels to pad the image (default=None).
            If padding is not None, pad image firstly with padding values.
            If a single number is provided, pad all borders with this value.
            If a tuple or list of 2 values are provided, pad the (left and top)
            with the first value and (right and bottom) with the second value.
            If 4 values are provided as a list or tuple,
            pad the left, top, right and bottom respectively.
        pad_if_needed (bool, optional): Pad the image if either side is smaller than
            the given output size (default=False).
        fill_value (Union[int, tuple], optional): The pixel intensity of the borders, only valid for
            padding_mode Border.CONSTANT. If it is a 3-tuple, it is used to fill R, G, B channels respectively.
            If it is an integer, it is used for all RGB channels.
            The fill_value values must be in range [0, 255] (default=0).
        padding_mode (Border mode, optional): The method of padding (default=Border.CONSTANT). It can be any of
            [Border.CONSTANT, Border.EDGE, Border.REFLECT, Border.SYMMETRIC].

            - Border.CONSTANT, means it fills the border with constant values.

            - Border.EDGE, means it pads with the last value on the edge.

            - Border.REFLECT, means it reflects the values on the edge omitting the last
              value of edge.

            - Border.SYMMETRIC, means it reflects the values on the edge repeating the last
              value of edge.

    Examples:
        >>> from mindspore.dataset.vision import Border
        >>> decode_op = c_vision.Decode()
        >>> random_crop_op = c_vision.RandomCrop(512, [200, 200, 200, 200], padding_mode=Border.EDGE)
        >>> transforms_list = [decode_op, random_crop_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_random_crop
    def __init__(self, size, padding=None, pad_if_needed=False, fill_value=0, padding_mode=Border.CONSTANT):
        if isinstance(size, int):
            size = (size, size)
        if padding is None:
            padding = (0, 0, 0, 0)
        else:
            padding = parse_padding(padding)
        if isinstance(fill_value, int):
            fill_value = tuple([fill_value] * 3)

        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill_value = fill_value
        self.padding_mode = padding_mode.value

    def parse(self):
        border_type = DE_C_BORDER_TYPE[self.padding_mode]
        return cde.RandomCropOperation(self.size, self.padding, self.pad_if_needed, self.fill_value, border_type)


class RandomCropDecodeResize(ImageTensorOperation):
    """
    A combination of `Crop`, `Decode` and `Resize`. It will get better performance for JPEG images.

    Args:
        size (Union[int, sequence]): The size of the output image.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).
        scale (tuple, optional): Range [min, max) of respective size of the
            original size to be cropped (default=(0.08, 1.0)).
        ratio (tuple, optional): Range [min, max) of aspect ratio to be
            cropped (default=(3. / 4., 4. / 3.)).
        interpolation (Inter mode, optional): Image interpolation mode (default=Inter.BILINEAR).
            It can be any of [Inter.BILINEAR, Inter.NEAREST, Inter.BICUBIC].

            - Inter.BILINEAR, means interpolation method is bilinear interpolation.

            - Inter.NEAREST, means interpolation method is nearest-neighbor interpolation.

            - Inter.BICUBIC, means interpolation method is bicubic interpolation.

        max_attempts (int, optional): The maximum number of attempts to propose a valid crop_area (default=10).
            If exceeded, fall back to use center_crop instead.

    Examples:
        >>> from mindspore.dataset.vision import Inter
        >>> resize_crop_decode_op = c_vision.RandomCropDecodeResize(size=(50, 75),
        ...                                                         scale=(0.25, 0.5),
        ...                                                         interpolation=Inter.NEAREST,
        ...                                                         max_attempts=5)
        >>> transforms_list = [resize_crop_decode_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_random_resize_crop
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation=Inter.BILINEAR, max_attempts=10):
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.max_attempts = max_attempts

    def parse(self):
        return cde.RandomCropDecodeResizeOperation(self.size, self.scale, self.ratio,
                                                   DE_C_INTER_MODE[self.interpolation],
                                                   self.max_attempts)

    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            raise TypeError("Input should be an encoded image in 1-D NumPy format, got {}.".format(type(img)))
        if img.ndim != 1 or img.dtype.type is not np.uint8:
            raise TypeError("Input should be an encoded image with uint8 type in 1-D NumPy format, " +
                            "got format:{}, dtype:{}.".format(type(img), img.dtype.type))
        return super().__call__(img)


class RandomCropWithBBox(ImageTensorOperation):
    """
    Crop the input image at a random location and adjust bounding boxes accordingly.

    Args:
        size (Union[int, sequence]): The output size of the cropped image.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).
        padding (Union[int, sequence], optional): The number of pixels to pad the image (default=None).
            If padding is not None, first pad image with padding values.
            If a single number is provided, pad all borders with this value.
            If a tuple or list of 2 values are provided, pad the (left and top)
            with the first value and (right and bottom) with the second value.
            If 4 values are provided as a list or tuple, pad the left, top, right and bottom respectively.
        pad_if_needed (bool, optional): Pad the image if either side is smaller than
            the given output size (default=False).
        fill_value (Union[int, tuple], optional): The pixel intensity of the borders, only valid for
            padding_mode Border.CONSTANT. If it is a 3-tuple, it is used to fill R, G, B channels respectively.
            If it is an integer, it is used for all RGB channels.
            The fill_value values must be in range [0, 255] (default=0).
        padding_mode (Border mode, optional): The method of padding (default=Border.CONSTANT). It can be any of
            [Border.CONSTANT, Border.EDGE, Border.REFLECT, Border.SYMMETRIC].

            - Border.CONSTANT, means it fills the border with constant values.

            - Border.EDGE, means it pads with the last value on the edge.

            - Border.REFLECT, means it reflects the values on the edge omitting the last
              value of edge.

            - Border.SYMMETRIC, means it reflects the values on the edge repeating the last
              value of edge.

    Examples:
        >>> decode_op = c_vision.Decode()
        >>> random_crop_with_bbox_op = c_vision.RandomCropWithBBox([512, 512], [200, 200, 200, 200])
        >>> transforms_list = [decode_op, random_crop_with_bbox_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_random_crop
    def __init__(self, size, padding=None, pad_if_needed=False, fill_value=0, padding_mode=Border.CONSTANT):
        if isinstance(size, int):
            size = (size, size)
        if padding is None:
            padding = (0, 0, 0, 0)
        else:
            padding = parse_padding(padding)

        if isinstance(fill_value, int):
            fill_value = tuple([fill_value] * 3)

        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill_value = fill_value
        self.padding_mode = padding_mode.value

    def parse(self):
        border_type = DE_C_BORDER_TYPE[self.padding_mode]
        return cde.RandomCropWithBBoxOperation(self.size, self.padding, self.pad_if_needed, self.fill_value,
                                               border_type)


class RandomHorizontalFlip(ImageTensorOperation):
    """
    Randomly flip the input image horizontally with a given probability.

    Args:
        prob (float, optional): Probability of the image being flipped (default=0.5).

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomHorizontalFlip(0.75)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_prob
    def __init__(self, prob=0.5):
        self.prob = prob

    def parse(self):
        return cde.RandomHorizontalFlipOperation(self.prob)


class RandomHorizontalFlipWithBBox(ImageTensorOperation):
    """
    Flip the input image horizontally, randomly with a given probability and adjust bounding boxes accordingly.

    Args:
        prob (float, optional): Probability of the image being flipped (default=0.5).

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomHorizontalFlipWithBBox(0.70)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_prob
    def __init__(self, prob=0.5):
        self.prob = prob

    def parse(self):
        return cde.RandomHorizontalFlipWithBBoxOperation(self.prob)


class RandomPosterize(ImageTensorOperation):
    """
    Reduce the number of bits for each color channel.

    Args:
        bits (sequence or int, optional): Range of random posterize to compress image.
            Bits values must be in range of [1,8], and include at
            least one integer value in the given range. It must be in
            (min, max) or integer format. If min=max, then it is a single fixed
            magnitude operation (default=(8, 8)).

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomPosterize((6, 8))]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_posterize
    def __init__(self, bits=(8, 8)):
        self.bits = bits

    def parse(self):
        bits = self.bits
        if isinstance(bits, int):
            bits = (bits, bits)
        return cde.RandomPosterizeOperation(bits)


class RandomResizedCrop(ImageTensorOperation):
    """
    Crop the input image to a random size and aspect ratio.

    Args:
        size (Union[int, sequence]): The size of the output image.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).
        scale (tuple, optional): Range [min, max) of respective size of the original
            size to be cropped (default=(0.08, 1.0)).
        ratio (tuple, optional): Range [min, max) of aspect ratio to be cropped
            (default=(3. / 4., 4. / 3.)).
        interpolation (Inter mode, optional): Image interpolation mode (default=Inter.BILINEAR).
            It can be any of [Inter.BILINEAR, Inter.NEAREST, Inter.BICUBIC].

            - Inter.BILINEAR, means interpolation method is bilinear interpolation.

            - Inter.NEAREST, means interpolation method is nearest-neighbor interpolation.

            - Inter.BICUBIC, means interpolation method is bicubic interpolation.

        max_attempts (int, optional): The maximum number of attempts to propose a valid
            crop_area (default=10). If exceeded, fall back to use center_crop instead.

    Examples:
        >>> from mindspore.dataset.vision import Inter
        >>> decode_op = c_vision.Decode()
        >>> resize_crop_op = c_vision.RandomResizedCrop(size=(50, 75), scale=(0.25, 0.5),
        ...                                             interpolation=Inter.BILINEAR)
        >>> transforms_list = [decode_op, resize_crop_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_random_resize_crop
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation=Inter.BILINEAR, max_attempts=10):
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.max_attempts = max_attempts

    def parse(self):
        return cde.RandomResizedCropOperation(self.size, self.scale, self.ratio, DE_C_INTER_MODE[self.interpolation],
                                              self.max_attempts)


class RandomResizedCropWithBBox(ImageTensorOperation):
    """
    Crop the input image to a random size and aspect ratio and adjust bounding boxes accordingly.

    Args:
        size (Union[int, sequence]): The size of the output image.
            If size is an integer, a square crop of size (size, size) is returned.
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
            crop area (default=10). If exceeded, fall back to use center crop instead.

    Examples:
        >>> from mindspore.dataset.vision import Inter
        >>> decode_op = c_vision.Decode()
        >>> bbox_op = c_vision.RandomResizedCropWithBBox(size=50, interpolation=Inter.NEAREST)
        >>> transforms_list = [decode_op, bbox_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_random_resize_crop
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation=Inter.BILINEAR, max_attempts=10):
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.max_attempts = max_attempts

    def parse(self):
        return cde.RandomResizedCropWithBBoxOperation(self.size, self.scale, self.ratio,
                                                      DE_C_INTER_MODE[self.interpolation], self.max_attempts)


class RandomResize(ImageTensorOperation):
    """
    Tensor operation to resize the input image using a randomly selected interpolation mode.

    Args:
        size (Union[int, sequence]): The output size of the resized image.
            If size is an integer, smaller edge of the image will be resized to this value with
            the same image aspect ratio.
            If size is a sequence of length 2, it should be (height, width).

    Examples:
        >>> # randomly resize image, keeping aspect ratio
        >>> transforms_list1 = [c_vision.Decode(), c_vision.RandomResize(50)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list1,
        ...                                                 input_columns=["image"])
        >>> # randomly resize image to landscape style
        >>> transforms_list2 = [c_vision.Decode(), c_vision.RandomResize((40, 60))]
        >>> image_folder_dataset_1 = image_folder_dataset_1.map(operations=transforms_list2,
        ...                                                     input_columns=["image"])
    """

    @check_resize
    def __init__(self, size):
        self.size = size

    def parse(self):
        size = self.size
        if isinstance(size, int):
            size = (size,)
        return cde.RandomResizeOperation(size)


class RandomResizeWithBBox(ImageTensorOperation):
    """
    Tensor operation to resize the input image using a randomly selected interpolation mode and adjust
    bounding boxes accordingly.

    Args:
        size (Union[int, sequence]): The output size of the resized image.
            If size is an integer, smaller edge of the image will be resized to this value with
            the same image aspect ratio.
            If size is a sequence of length 2, it should be (height, width).

    Examples:
        >>> # randomly resize image with bounding boxes, keeping aspect ratio
        >>> transforms_list1 = [c_vision.Decode(), c_vision.RandomResizeWithBBox(60)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list1,
        ...                                                 input_columns=["image"])
        >>> # randomly resize image with bounding boxes to portrait style
        >>> transforms_list2 = [c_vision.Decode(), c_vision.RandomResizeWithBBox((80, 60))]
        >>> image_folder_dataset_1 = image_folder_dataset_1.map(operations=transforms_list2,
        ...                                                     input_columns=["image"])
    """

    @check_resize
    def __init__(self, size):
        self.size = size

    def parse(self):
        size = self.size
        if isinstance(size, int):
            size = (size,)
        return cde.RandomResizeWithBBoxOperation(size)


class RandomRotation(ImageTensorOperation):
    """
    Rotate the input image by a random angle.

    Args:
        degrees (Union[int, float, sequence): Range of random rotation degrees.
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
        fill_value (Union[int, tuple], optional): Optional fill color for the area outside the rotated image.
            If it is a 3-tuple, it is used to fill R, G, B channels respectively.
            If it is an integer, it is used for all RGB channels.
            The fill_value values must be in range [0, 255] (default=0).

    Examples:
        >>> from mindspore.dataset.vision import Inter
        >>> transforms_list = [c_vision.Decode(),
        ...                    c_vision.RandomRotation(degrees=5.0,
        ...                    resample=Inter.NEAREST,
        ...                    expand=True)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_random_rotation
    def __init__(self, degrees, resample=Inter.NEAREST, expand=False, center=None, fill_value=0):
        if isinstance(degrees, numbers.Number):
            degrees = degrees % 360
        if isinstance(degrees, (list, tuple)):
            degrees = [degrees[0] % 360, degrees[1] % 360]
            if degrees[0] > degrees[1]:
                degrees[1] += 360

        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill_value = fill_value

    def parse(self):
        # pylint false positive
        # pylint: disable=E1130
        degrees = (-self.degrees, self.degrees) if isinstance(self.degrees, numbers.Number) else self.degrees
        interpolation = DE_C_INTER_MODE[self.resample]
        expand = self.expand
        center = (-1, -1) if self.center is None else self.center
        fill_value = tuple([self.fill_value] * 3) if isinstance(self.fill_value, int) else self.fill_value
        return cde.RandomRotationOperation(degrees, interpolation, expand, center, fill_value)


class RandomSelectSubpolicy(ImageTensorOperation):
    """
    Choose a random sub-policy from a list to be applied on the input image. A sub-policy is a list of tuples
    (op, prob), where op is a TensorOp operation and prob is the probability that this op will be applied. Once
    a sub-policy is selected, each op within the subpolicy with be applied in sequence according to its probability.

    Args:
        policy (list(list(tuple(TensorOp, float))): List of sub-policies to choose from.

    Examples:
        >>> policy = [[(c_vision.RandomRotation((45, 45)), 0.5),
        ...            (c_vision.RandomVerticalFlip(), 1),
        ...            (c_vision.RandomColorAdjust(), 0.8)],
        ...           [(c_vision.RandomRotation((90, 90)), 1),
        ...            (c_vision.RandomColorAdjust(), 0.2)]]
        >>> image_folder_dataset = image_folder_dataset.map(operations=c_vision.RandomSelectSubpolicy(policy),
        ...                                                 input_columns=["image"])
    """

    @check_random_select_subpolicy_op
    def __init__(self, policy):
        self.policy = policy

    def parse(self):
        """
        Return a C++ representation of the operator for execution
        """
        policy = []
        for list_one in self.policy:
            policy_one = []
            for list_two in list_one:
                if list_two[0] and getattr(list_two[0], 'parse', None):
                    policy_one.append((list_two[0].parse(), list_two[1]))
                else:
                    policy_one.append((list_two[0], list_two[1]))
            policy.append(policy_one)
        return cde.RandomSelectSubpolicyOperation(policy)


class RandomSharpness(ImageTensorOperation):
    """
    Adjust the sharpness of the input image by a fixed or random degree. Degree of 0.0 gives a blurred image,
    degree of 1.0 gives the original image, and degree of 2.0 gives a sharpened image.

    Args:
        degrees (Union[list, tuple], optional): Range of random sharpness adjustment degrees. It should be in
            (min, max) format. If min=max, then it is a single fixed magnitude operation (default = (0.1, 1.9)).

    Raises:
        TypeError : If degrees is not a list or tuple.
        ValueError: If degrees is negative.
        ValueError: If degrees is in (max, min) format instead of (min, max).

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomSharpness(degrees=(0.2, 1.9))]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_positive_degrees
    def __init__(self, degrees=(0.1, 1.9)):
        self.degrees = degrees

    def parse(self):
        return cde.RandomSharpnessOperation(self.degrees)


class RandomSolarize(ImageTensorOperation):
    """
    Randomly invert the pixel values of input image within given range.

    Args:
        threshold (tuple, optional): Range of random solarize threshold (default=(0, 255)).
            Threshold values should always be in (min, max) format,
            where min <= max, min and max are integers in the range (0, 255).
            If min=max, then invert all pixel values above min(max).

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomSolarize(threshold=(10,100))]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_random_solarize
    def __init__(self, threshold=(0, 255)):
        self.threshold = threshold

    def parse(self):
        return cde.RandomSolarizeOperation(self.threshold)


class RandomVerticalFlip(ImageTensorOperation):
    """
    Randomly flip the input image vertically with a given probability.

    Args:
        prob (float, optional): Probability of the image being flipped (default=0.5).

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomVerticalFlip(0.25)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_prob
    def __init__(self, prob=0.5):
        self.prob = prob

    def parse(self):
        return cde.RandomVerticalFlipOperation(self.prob)


class RandomVerticalFlipWithBBox(ImageTensorOperation):
    """
    Flip the input image vertically, randomly with a given probability and adjust bounding boxes accordingly.

    Args:
        prob (float, optional): Probability of the image being flipped (default=0.5).

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomVerticalFlipWithBBox(0.20)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_prob
    def __init__(self, prob=0.5):
        self.prob = prob

    def parse(self):
        return cde.RandomVerticalFlipWithBBoxOperation(self.prob)


class Rescale(ImageTensorOperation):
    """
    Tensor operation to rescale the input image.

    Args:
        rescale (float): Rescale factor.
        shift (float): Shift factor.

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.Rescale(1.0 / 255.0, -1.0)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_rescale
    def __init__(self, rescale, shift):
        self.rescale = rescale
        self.shift = shift

    def parse(self):
        return cde.RescaleOperation(self.rescale, self.shift)


class Resize(ImageTensorOperation):
    """
    Resize the input image to the given size.

    Args:
        size (Union[int, sequence]): The output size of the resized image.
            If size is an integer, the smaller edge of the image will be resized to this value with
            the same image aspect ratio.
            If size is a sequence of length 2, it should be (height, width).
        interpolation (Inter mode, optional): Image interpolation mode (default=Inter.LINEAR).
            It can be any of [Inter.LINEAR, Inter.NEAREST, Inter.BICUBIC].

            - Inter.LINEAR, means interpolation method is bilinear interpolation.

            - Inter.NEAREST, means interpolation method is nearest-neighbor interpolation.

            - Inter.BICUBIC, means interpolation method is bicubic interpolation.

            - Inter.AREA, means interpolation method is pixel area interpolation.

    Examples:
        >>> from mindspore.dataset.vision import Inter
        >>> decode_op = c_vision.Decode()
        >>> resize_op = c_vision.Resize([100, 75], Inter.BICUBIC)
        >>> transforms_list = [decode_op, resize_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_resize_interpolation
    def __init__(self, size, interpolation=Inter.LINEAR):
        if isinstance(size, int):
            size = (size,)
        self.size = size
        self.interpolation = interpolation

    def parse(self):
        return cde.ResizeOperation(self.size, DE_C_INTER_MODE[self.interpolation])


class ResizeWithBBox(ImageTensorOperation):
    """
    Resize the input image to the given size and adjust bounding boxes accordingly.

    Args:
        size (Union[int, sequence]): The output size of the resized image.
            If size is an integer, smaller edge of the image will be resized to this value with
            the same image aspect ratio.
            If size is a sequence of length 2, it should be (height, width).
        interpolation (Inter mode, optional): Image interpolation mode (default=Inter.LINEAR).
            It can be any of [Inter.LINEAR, Inter.NEAREST, Inter.BICUBIC].

            - Inter.LINEAR, means interpolation method is bilinear interpolation.

            - Inter.NEAREST, means interpolation method is nearest-neighbor interpolation.

            - Inter.BICUBIC, means interpolation method is bicubic interpolation.

    Examples:
        >>> from mindspore.dataset.vision import Inter
        >>> decode_op = c_vision.Decode()
        >>> bbox_op = c_vision.ResizeWithBBox(50, Inter.NEAREST)
        >>> transforms_list = [decode_op, bbox_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_resize_interpolation
    def __init__(self, size, interpolation=Inter.LINEAR):
        self.size = size
        self.interpolation = interpolation

    def parse(self):
        size = self.size
        if isinstance(size, int):
            size = (size,)
        return cde.ResizeWithBBoxOperation(size, DE_C_INTER_MODE[self.interpolation])


class SoftDvppDecodeRandomCropResizeJpeg(ImageTensorOperation):
    """
    Tensor operation to decode, random crop and resize JPEG image using the simulation algorithm of
    Ascend series chip DVPP module.

    The usage scenario is consistent with SoftDvppDecodeResizeJpeg.
    The input image size should be in range [32*32, 8192*8192].
    The zoom-out and zoom-in multiples of the image length and width should in the range [1/32, 16].
    Only images with an even resolution can be output. The output of odd resolution is not supported.

    Args:
        size (Union[int, sequence]): The size of the output image.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).
        scale (tuple, optional): Range [min, max) of respective size of the
            original size to be cropped (default=(0.08, 1.0)).
        ratio (tuple, optional): Range [min, max) of aspect ratio to be
            cropped (default=(3. / 4., 4. / 3.)).
        max_attempts (int, optional): The maximum number of attempts to propose a valid crop_area (default=10).
            If exceeded, fall back to use center_crop instead.

    Examples:
        >>> # decode, randomly crop and resize image, keeping aspect ratio
        >>> transforms_list1 = [c_vision.Decode(), c_vision.SoftDvppDecodeRandomCropResizeJpeg(90)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list1,
        ...                                                 input_columns=["image"])
        >>> # decode, randomly crop and resize to landscape style
        >>> transforms_list2 = [c_vision.Decode(), c_vision.SoftDvppDecodeRandomCropResizeJpeg((80, 100))]
        >>> image_folder_dataset_1 = image_folder_dataset_1.map(operations=transforms_list2,
        ...                                                     input_columns=["image"])
    """

    @check_soft_dvpp_decode_random_crop_resize_jpeg
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), max_attempts=10):
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.max_attempts = max_attempts

    def parse(self):
        return cde.SoftDvppDecodeRandomCropResizeJpegOperation(self.size, self.scale, self.ratio, self.max_attempts)


class SoftDvppDecodeResizeJpeg(ImageTensorOperation):
    """
    Tensor operation to decode and resize JPEG image using the simulation algorithm of
    Ascend series chip DVPP module.

    It is recommended to use this algorithm in the following scenarios:
    When training, the DVPP of the Ascend chip is not used,
    and the DVPP of the Ascend chip is used during inference,
    and the accuracy of inference is lower than the accuracy of training;
    and the input image size should be in range [32*32, 8192*8192].
    The zoom-out and zoom-in multiples of the image length and width should in the range [1/32, 16].
    Only images with an even resolution can be output. The output of odd resolution is not supported.

    Args:
        size (Union[int, sequence]): The output size of the resized image.
            If size is an integer, smaller edge of the image will be resized to this value with
            the same image aspect ratio.
            If size is a sequence of length 2, it should be (height, width).

    Examples:
        >>> # decode and resize image, keeping aspect ratio
        >>> transforms_list1 = [c_vision.Decode(), c_vision.SoftDvppDecodeResizeJpeg(70)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list1,
        ...                                                 input_columns=["image"])
        >>> # decode and resize to portrait style
        >>> transforms_list2 = [c_vision.Decode(), c_vision.SoftDvppDecodeResizeJpeg((80, 60))]
        >>> image_folder_dataset_1 = image_folder_dataset_1.map(operations=transforms_list2,
        ...                                                     input_columns=["image"])
    """

    @check_resize
    def __init__(self, size):
        if isinstance(size, int):
            size = (size,)
        self.size = size

    def parse(self):
        return cde.SoftDvppDecodeResizeJpegOperation(self.size)


class UniformAugment(ImageTensorOperation):
    """
    Tensor operation to perform randomly selected augmentation.

    Args:
        transforms: List of C++ operations (Python operations are not accepted).
        num_ops (int, optional): Number of operations to be selected and applied (default=2).

    Examples:
        >>> import mindspore.dataset.vision.py_transforms as py_vision
        >>> transforms_list = [c_vision.RandomHorizontalFlip(),
        ...                    c_vision.RandomVerticalFlip(),
        ...                    c_vision.RandomColorAdjust(),
        ...                    c_vision.RandomRotation(degrees=45)]
        >>> uni_aug_op = c_vision.UniformAugment(transforms=transforms_list, num_ops=2)
        >>> transforms_all = [c_vision.Decode(), c_vision.Resize(size=[224, 224]),
        ...                   uni_aug_op, py_vision.ToTensor()]
        >>> image_folder_dataset_1 = image_folder_dataset.map(operations=transforms_all,
        ...                                                   input_columns="image",
        ...                                                   num_parallel_workers=1)
    """

    @check_uniform_augment_cpp
    def __init__(self, transforms, num_ops=2):
        self.transforms = transforms
        self.num_ops = num_ops

    def parse(self):
        transforms = []
        for op in self.transforms:
            if op and getattr(op, 'parse', None):
                transforms.append(op.parse())
            else:
                transforms.append(op)
        return cde.UniformAugOperation(transforms, self.num_ops)
