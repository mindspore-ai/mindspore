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
# ==============================================================================
"""
The module vision.transforms provides many kinds of image augmentation methods
and image-related conversion methods
(e.g. including with PIL.Image.Image and numpy.ndarray).
to perform various computer vision tasks.
Users can apply suitable augmentations on image data
to improve their training models.
Users can also self-define their own augmentation methods with Python Pillow (PIL)

For the different methods in this module, implementation is based in C++ and/or Python.
The C++ implementation is inherited from mindspore._c_dataengine, provides high performance
and is mainly based on OpenCV.
The Python implementation is mainly based on PIL.

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
    >>> transforms_list = [vision.Decode(),
    ...                    vision.Resize((256, 256), interpolation=Inter.LINEAR),
    ...                    vision.RandomCrop(200, padding_mode=Border.EDGE),
    ...                    vision.RandomRotation((0, 15)),
    ...                    vision.Normalize((100, 115.0, 121.0), (71.0, 68.0, 70.0)),
    ...                    vision.HWC2CHW()]
    >>> onehot_op = data_transforms.OneHot(num_classes=10)
    >>> # apply the transformation to the dataset through data1.map()
    >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
    ...                                                 input_columns="image")
    >>> image_folder_dataset = image_folder_dataset.map(operations=onehot_op,
    ...                                                 input_columns="label")
"""
import numbers
import random
import numpy as np

import mindspore._c_dataengine as cde
from . import py_transforms_util as util
from .py_transforms_util import is_pil
from .utils import Border, Inter, parse_padding
from .utils import ImageBatchFormat, ConvertMode, SliceMode, AutoAugmentPolicy
from .validators import check_adjust_gamma, check_alpha, check_auto_contrast, check_center_crop, check_cutout_new, \
    check_five_crop, check_hsv_to_rgb, check_linear_transform, check_mix_up, check_normalize, \
    check_normalizepad, check_num_channels, check_pad, check_positive_degrees, check_prob, check_random_affine, \
    check_random_color_adjust, check_random_crop, check_random_erasing, check_random_perspective, \
    check_random_resize_crop, check_random_rotation, check_resize_interpolation, check_rgb_to_hsv, \
    check_ten_crop, check_crop, check_mix_up_batch_c, check_range, \
    check_resize, check_rescale, check_uniform_augment_cpp, check_convert_color, check_random_auto_contrast, \
    check_random_adjust_sharpness, check_auto_augment, \
    check_bounding_box_augment_cpp, check_random_select_subpolicy_op, check_random_solarize, \
    check_soft_dvpp_decode_random_crop_resize_jpeg, FLOAT_MAX_INTEGER, \
    check_cut_mix_batch_c, check_posterize, check_gaussian_blur, check_rotate, check_slice_patches
from ..transforms.py_transforms_util import Implementation
from ..transforms.transforms import TensorOperation, PyTensorOperation, CompoundOperation, TypeCast


class AdjustGamma(TensorOperation, PyTensorOperation):
    r"""
    Apply gamma correction on input image. Input image is expected to be in [..., H, W, C] or [H, W] format.
    .. math::
        I_{\text{out}} = 255 \times \text{gain} \times \left(\frac{I_{\text{in}}}{255}\right)^{\gamma}

    See `Gamma Correction`_ for more details.

    .. _Gamma Correction: https://en.wikipedia.org/wiki/Gamma_correction

    Args:
        gamma (float): Non negative real number.
            The output image pixel value is exponentially related to the input image pixel value.
            gamma larger than 1 make the shadows darker,
            while gamma smaller than 1 make dark regions lighter.
        gain (float, optional): The constant multiplier (default=1).

    Raises:
        TypeError: If `gain` is not of type float.
        TypeError: If `gamma` is not of type float.
        ValueError: If `gamma` is less than 0.
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [vision.Decode(), vision.AdjustGamma(gamma=10.0, gain=1.0)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_adjust_gamma
    def __init__(self, gamma, gain=1):
        super(AdjustGamma, self).__init__()
        self.gamma = gamma
        self.gain = gain
        self.random = False

    def parse(self):
        return cde.AdjustGammaOperation(self.gamma, self.gain)

    def execute_pil(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be gamma adjusted.

        Returns:
            PIL Image, gamma adjusted image.
        """
        return util.adjust_gamma(img, self.gamma, self.gain)


class AutoAugment(TensorOperation):
    """
    Apply AutoAugment data augmentation method based on
    `AutoAugment: Learning Augmentation Strategies from Data <https://arxiv.org/pdf/1805.09501.pdf>`_.
    This operation works only with 3-channel RGB images.

    Args:
        policy (AutoAugmentPolicy, optional): AutoAugment policies learned on different datasets
            (default=AutoAugmentPolicy.IMAGENET).
            It can be any of [AutoAugmentPolicy.IMAGENET, AutoAugmentPolicy.CIFAR10, AutoAugmentPolicy.SVHN].
            Randomly apply 2 operations from a candidate set. See auto augmentation details in AutoAugmentPolicy.

            - AutoAugmentPolicy.IMAGENET, means to apply AutoAugment learned on ImageNet dataset.

            - AutoAugmentPolicy.CIFAR10, means to apply AutoAugment learned on Cifar10 dataset.

            - AutoAugmentPolicy.SVHN, means to apply AutoAugment learned on SVHN dataset.

        interpolation (Inter, optional): Image interpolation mode for Resize operator (default=Inter.NEAREST).
            It can be any of [Inter.NEAREST, Inter.BILINEAR, Inter.BICUBIC, Inter.AREA].

            - Inter.NEAREST: means interpolation method is nearest-neighbor interpolation.

            - Inter.BILINEAR: means interpolation method is bilinear interpolation.

            - Inter.BICUBIC: means the interpolation method is bicubic interpolation.

            - Inter.AREA: means the interpolation method is area interpolation.

        fill_value (Union[int, tuple], optional): Pixel fill value for the area outside the transformed image.
            It can be an int or a 3-tuple. If it is a 3-tuple, it is used to fill R, G, B channels respectively.
            If it is an integer, it is used for all RGB channels. The fill_value values must be in range [0, 255]
            (default=0).

    Raises:
        TypeError: If `policy` is not of type AutoAugmentPolicy.
        TypeError: If `interpolation` is not of type Inter.
        TypeError: If `fill_value` is not an integer or a tuple of length 3.
        RuntimeError: If given tensor shape is not <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.vision import AutoAugmentPolicy, Inter

        >>> transforms_list = [vision.Decode(), vision.AutoAugment(policy=AutoAugmentPolicy.IMAGENET,
        ...                                                        interpolation=Inter.NEAREST,
        ...                                                        fill_value=0)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_auto_augment
    def __init__(self, policy=AutoAugmentPolicy.IMAGENET, interpolation=Inter.NEAREST, fill_value=0):
        super().__init__()
        self.policy = policy
        self.interpolation = interpolation
        if isinstance(fill_value, int):
            fill_value = tuple([fill_value] * 3)
        self.fill_value = fill_value
        self.implementation = Implementation.C

    def parse(self):
        return cde.AutoAugmentOperation(AutoAugmentPolicy.to_c_type(self.policy), Inter.to_c_type(self.interpolation),
                                        self.fill_value)


class AutoContrast(TensorOperation, PyTensorOperation):
    """
    Apply automatic contrast on input image. This operator calculates histogram of image, reassign cutoff percent
    of the lightest pixels from histogram to 255, and reassign cutoff percent of the darkest pixels from histogram to 0.

    Args:
        cutoff (float, optional): Percent of lightest and darkest pixels to cut off from
            the histogram of input image. The value must be in the range [0.0, 50.0) (default=0.0).
        ignore (Union[int, sequence], optional): The background pixel values to ignore,
            The ignore values must be in range [0, 255] (default=None).

    Raises:
        TypeError: If `cutoff` is not of type float.
        TypeError: If `ignore` is not of type int or sequence.
        ValueError: If `cutoff` is not in range [0, 50.0).
        ValueError: If `ignore` is not in range [0, 255].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [vision.Decode(), vision.AutoContrast(cutoff=10.0, ignore=[10, 20])]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_auto_contrast
    def __init__(self, cutoff=0.0, ignore=None):
        super().__init__()
        if ignore is None:
            ignore = []
        if isinstance(ignore, int):
            ignore = [ignore]
        self.cutoff = cutoff
        self.ignore = ignore
        self.random = False

    def parse(self):
        return cde.AutoContrastOperation(self.cutoff, self.ignore)

    def execute_pil(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be automatically contrasted.

        Returns:
            PIL Image, automatically contrasted image.
        """
        return util.auto_contrast(img, self.cutoff, self.ignore)


class BoundingBoxAugment(TensorOperation):
    """
    Apply a given image processing operation on a random selection of bounding box regions of a given image.

    Args:
        transform (TensorOperation): C++ transformation operation to be applied on random selection
            of bounding box regions of a given image.
        ratio (float, optional): Ratio of bounding boxes to apply augmentation on.
            Range: [0, 1] (default=0.3).

    Raises:
        TypeError: If `transform` is an image processing operation in :class:`mindspore.dataset.vision.transforms`.
        TypeError: If `ratio` is not of type float.
        ValueError: If `ratio` is not in range [0, 1].
        RuntimeError: If given bounding box is invalid.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> # set bounding box operation with ratio of 1 to apply rotation on all bounding boxes
        >>> bbox_aug_op = vision.BoundingBoxAugment(vision.RandomRotation(90), 1)
        >>> # map to apply ops
        >>> image_folder_dataset = image_folder_dataset.map(operations=[bbox_aug_op],
        ...                                                 input_columns=["image", "bbox"],
        ...                                                 output_columns=["image", "bbox"],
        ...                                                 column_order=["image", "bbox"])
    """

    @check_bounding_box_augment_cpp
    def __init__(self, transform, ratio=0.3):
        super().__init__()
        self.ratio = ratio
        self.transform = transform
        self.implementation = Implementation.C

    def parse(self):
        if self.transform and getattr(self.transform, 'parse', None):
            transform = self.transform.parse()
        else:
            transform = self.transform
        return cde.BoundingBoxAugmentOperation(transform, self.ratio)


class CenterCrop(TensorOperation, PyTensorOperation):
    """
    Crop the input image at the center to the given size. If input image size is smaller than output size,
    input image will be padded with 0 before cropping.

    Args:
        size (Union[int, sequence]): The output size of the cropped image.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).
            The size value(s) must be larger than 0.

    Raises:
        TypeError: If `size` is not of type integer or sequence.
        ValueError: If `size` is less than or equal to 0.
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> # crop image to a square
        >>> transforms_list1 = [vision.Decode(), vision.CenterCrop(50)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list1,
        ...                                                 input_columns=["image"])
        >>> # crop image to portrait style
        >>> transforms_list2 = [vision.Decode(), vision.CenterCrop((60, 40))]
        >>> image_folder_dataset_1 = image_folder_dataset_1.map(operations=transforms_list2,
        ...                                                     input_columns=["image"])
    """

    @check_center_crop
    def __init__(self, size):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.random = False

    def parse(self):
        return cde.CenterCropOperation(self.size)

    def execute_pil(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be center cropped.

        Returns:
            PIL Image, cropped image.
        """
        return util.center_crop(img, self.size)


class ConvertColor(TensorOperation):
    """
    Change the color space of the image.

    Args:
        convert_mode (ConvertMode): The mode of image channel conversion.

            - ConvertMode.COLOR_BGR2BGRA, Convert BGR image to BGRA image.

            - ConvertMode.COLOR_RGB2RGBA, Convert RGB image to RGBA image.

            - ConvertMode.COLOR_BGRA2BGR, Convert BGRA image to BGR image.

            - ConvertMode.COLOR_RGBA2RGB, Convert RGBA image to RGB image.

            - ConvertMode.COLOR_BGR2RGBA, Convert BGR image to RGBA image.

            - ConvertMode.COLOR_RGB2BGRA, Convert RGB image to BGRA image.

            - ConvertMode.COLOR_RGBA2BGR, Convert RGBA image to BGR image.

            - ConvertMode.COLOR_BGRA2RGB, Convert BGRA image to RGB image.

            - ConvertMode.COLOR_BGR2RGB, Convert BGR image to RGB image.

            - ConvertMode.COLOR_RGB2BGR, Convert RGB image to BGR image.

            - ConvertMode.COLOR_BGRA2RGBA, Convert BGRA image to RGBA image.

            - ConvertMode.COLOR_RGBA2BGRA, Convert RGBA image to BGRA image.

            - ConvertMode.COLOR_BGR2GRAY, Convert BGR image to GRAY image.

            - ConvertMode.COLOR_RGB2GRAY, Convert RGB image to GRAY image.

            - ConvertMode.COLOR_GRAY2BGR, Convert GRAY image to BGR image.

            - ConvertMode.COLOR_GRAY2RGB, Convert GRAY image to RGB image.

            - ConvertMode.COLOR_GRAY2BGRA, Convert GRAY image to BGRA image.

            - ConvertMode.COLOR_GRAY2RGBA, Convert GRAY image to RGBA image.

            - ConvertMode.COLOR_BGRA2GRAY, Convert BGRA image to GRAY image.

            - ConvertMode.COLOR_RGBA2GRAY, Convert RGBA image to GRAY image.

    Raises:
        TypeError: If `convert_mode` is not of type :class:`mindspore.dataset.vision.transforms.ConvertMode`.
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import mindspore.dataset.vision.utils as mode
        >>> # Convert RGB images to GRAY images
        >>> convert_op = vision.ConvertColor(mode.ConvertMode.COLOR_RGB2GRAY)
        >>> image_folder_dataset = image_folder_dataset.map(operations=convert_op,
        ...                                                 input_columns=["image"])
        >>> # Convert RGB images to BGR images
        >>> convert_op = vision.ConvertColor(mode.ConvertMode.COLOR_RGB2BGR)
        >>> image_folder_dataset_1 = image_folder_dataset_1.map(operations=convert_op,
        ...                                                     input_columns=["image"])
    """

    @check_convert_color
    def __init__(self, convert_mode):
        super().__init__()
        self.convert_mode = convert_mode
        self.implementation = Implementation.C

    def parse(self):
        return cde.ConvertColorOperation(ConvertMode.to_c_type(self.convert_mode))


class Crop(TensorOperation):
    """
    Crop the input image at a specific location.

    Args:
        coordinates(sequence): Coordinates of the upper left corner of the cropping image. Must be a sequence of two
            values, in the form of (top, left).
        size (Union[int, sequence]): The output size of the cropped image.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).
            The size value(s) must be larger than 0.

    Raises:
        TypeError: If `coordinates` is not of type sequence.
        TypeError: If `size` is not of type integer or sequence.
        ValueError: If `coordinates` is less than 0.
        ValueError: If `size` is less than or equal to 0.
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> decode_op = vision.Decode()
        >>> crop_op = vision.Crop((0, 0), 32)
        >>> transforms_list = [decode_op, crop_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_crop
    def __init__(self, coordinates, size):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.coordinates = coordinates
        self.size = size
        self.implementation = Implementation.C

    def parse(self):
        return cde.CropOperation(self.coordinates, self.size)


class CutMixBatch(TensorOperation):
    """
    Apply CutMix transformation on input batch of images and labels.
    Note that you need to make labels into one-hot format and batched before calling this operator.

    Args:
        image_batch_format (ImageBatchFormat): The method of padding. Can be any of
            [ImageBatchFormat.NHWC, ImageBatchFormat.NCHW].
        alpha (float, optional): Hyperparameter of beta distribution, must be larger than 0 (default = 1.0).
        prob (float, optional): The probability by which CutMix is applied to each image, range: [0, 1] (default = 1.0).

    Raises:
        TypeError: If `image_batch_format` is not of type :class:`mindspore.dataset.vision.ImageBatchFormat`.
        TypeError: If `alpha` is not of type float.
        TypeError: If `prob` is not of type float.
        ValueError: If `alpha` is less than or equal 0.
        ValueError: If `prob` is not in range [0, 1].
        RuntimeError: If given tensor shape is not <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.vision import ImageBatchFormat
        >>> onehot_op = data_transforms.OneHot(num_classes=10)
        >>> image_folder_dataset= image_folder_dataset.map(operations=onehot_op,
        ...                                                input_columns=["label"])
        >>> cutmix_batch_op = vision.CutMixBatch(ImageBatchFormat.NHWC, 1.0, 0.5)
        >>> image_folder_dataset = image_folder_dataset.batch(5)
        >>> image_folder_dataset = image_folder_dataset.map(operations=cutmix_batch_op,
        ...                                                 input_columns=["image", "label"])
    """

    @check_cut_mix_batch_c
    def __init__(self, image_batch_format, alpha=1.0, prob=1.0):
        super().__init__()
        self.image_batch_format = image_batch_format.value
        self.alpha = alpha
        self.prob = prob
        self.implementation = Implementation.C

    def parse(self):
        return cde.CutMixBatchOperation(ImageBatchFormat.to_c_type(self.image_batch_format), self.alpha, self.prob)


class CutOut(TensorOperation):
    """
    Randomly cut (mask) out a given number of square patches from the input image array.

    Args:
        length (int): The side length of each square patch, must be larger than 0.
        num_patches (int, optional): Number of patches to be cut out of an image, must be larger than 0. (default=1).
        is_hwc (bool): Whether the input image is in HWC format. True - HWC format, False - CHW format. Default: True.

    Raises:
        TypeError: If `length` is not of type integer.
        TypeError: If `num_patches` is not of type integer.
        ValueError: If `length` is less than or equal 0.
        ValueError: If `num_patches` is less than or equal 0.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [vision.Decode(), vision.CutOut(80, num_patches=10)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_cutout_new
    def __init__(self, length, num_patches=1, is_hwc=True):
        super().__init__()
        self.length = length
        self.num_patches = num_patches
        self.is_hwc = is_hwc
        self.random = False
        self.implementation = Implementation.C

    def parse(self):
        return cde.CutOutOperation(self.length, self.num_patches, self.is_hwc)


class Decode(TensorOperation, PyTensorOperation):
    """
    Decode the input image in RGB mode.

    Args:
        to_pil (bool, optional): decode to PIL Image (default=False).

    Raises:
        RuntimeError: If given tensor is not a 1D sequence.
        ValueError: If the input is not raw image bytes.
        ValueError: If the input image is already decoded.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [vision.Decode(), vision.RandomHorizontalFlip()]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    def __init__(self, to_pil=False):
        super().__init__()
        self.to_pil = to_pil
        if to_pil:
            self.implementation = Implementation.PY
        else:
            self.implementation = Implementation.C

    def __call__(self, img):
        """
        Call method.

        Args:
            img (NumPy): Image to be decoded.

        Returns:
            img (NumPy, PIL Image), Decoded image.
        """
        if self.implementation == Implementation.PY:
            return util.decode(img)

        if isinstance(img, bytes):
            img = np.frombuffer(img, np.uint8)
        elif not isinstance(img, np.ndarray) or img.ndim != 1 or img.dtype.type is np.str_:
            raise TypeError(
                "Input should be an encoded image in 1-D NumPy format, got {}.".format(type(img)))
        return super().__call__(img)

    def parse(self):
        return cde.DecodeOperation(True)


class Equalize(TensorOperation, PyTensorOperation):
    """
    Apply histogram equalization on input image.

    Raises:
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [vision.Decode(), vision.Equalize()]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    def __init__(self):
        super().__init__()
        self.random = False

    def parse(self):
        return cde.EqualizeOperation()

    def execute_pil(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be equalized.

        Returns:
            PIL Image, equalized image.
        """

        return util.equalize(img)


class FiveCrop(PyTensorOperation):
    """
    Crop the given image into one central crop and four corners.

    Args:
        size (Union[int, sequence]): The size of the cropped image.
            If a single integer is provided, a square of size (size, size) will be cropped with this value.
            If a sequence of length 2 is provided, an image of size (height, width) will be cropped.

    Raises:
        TypeError: If `size` is not of type integer or sequence of integer.
        ValueError: If `size` is not positive.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy
        >>> from mindspore.dataset.transforms.transforms import Compose
        >>>
        >>> transforms_list = Compose([vision.Decode(to_pil=True),
        ...                            vision.FiveCrop(size=200),
        ...                            # 4D stack of 5 images
        ...                            lambda *images: numpy.stack([vision.ToTensor()(image) for image in images])])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_five_crop
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.random = False
        self.implementation = Implementation.PY

    def execute_pil(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            tuple, a tuple of five PIL Image in order of top_left, top_right, bottom_left, bottom_right, center.
        """
        return util.five_crop(img, self.size)


class GaussianBlur(TensorOperation):
    """
    Blur input image with the specified Gaussian kernel.

    Args:
        kernel_size (Union[int, Sequence[int]]): Size of the Gaussian kernel to use. The value must be positive and odd.
            If only an integer is provided, the kernel size will be (kernel_size, kernel_size). If a sequence of integer
            is provided, it must be a sequence of 2 values which represents (width, height).
        sigma (Union[float, Sequence[float]], optional): Standard deviation of the Gaussian kernel to use
            (default=None). The value must be positive. If only a float is provided, the sigma will be (sigma, sigma).
            If a sequence of float is provided, it must be a sequence of 2 values which represents (width, height).
            If None is provided, the sigma will be calculated as ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8.

    Raises:
        TypeError: If `kernel_size` is not of type int or Sequence[int].
        TypeError: If `sigma` is not of type float or Sequence[float].
        ValueError: If `kernel_size` is not positive and odd.
        ValueError: If `sigma` is not positive.
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [vision.Decode(to_pil=True)), vision.GaussianBlur(3, 3)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_gaussian_blur
    def __init__(self, kernel_size, sigma=None):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)
        if sigma is None:
            sigma = (0,)
        elif isinstance(sigma, (int, float)):
            sigma = (float(sigma),)
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.implementation = Implementation.C

    def parse(self):
        return cde.GaussianBlurOperation(self.kernel_size, self.sigma)


class Grayscale(PyTensorOperation):
    """
    Convert the input PIL Image to grayscale.

    Args:
        num_output_channels (int): The number of channels desired for the output image, must be 1 or 3.
            If 3 is provided, the returned image will have 3 identical RGB channels. Default: 1.

    Raises:
        TypeError: If `num_output_channels` is not of type integer.
        ValueError: If `num_output_channels` is not 1 or 3.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.transforms import Compose
        >>>
        >>> transforms_list = Compose([vision.Decode(to_pil=True)),
        ...                            vision.Grayscale(3),
        ...                            vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_num_channels
    def __init__(self, num_output_channels=1):
        super().__init__()
        self.num_output_channels = num_output_channels
        self.random = False
        self.implementation = Implementation.PY

    def execute_pil(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image, converted grayscale image.
        """
        return util.grayscale(img, num_output_channels=self.num_output_channels)


class HorizontalFlip(TensorOperation):
    """
    Flip the input image horizontally.

    Raises:
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [vision.Decode(to_pil=True)), vision.HorizontalFlip()]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    def __init__(self):
        super().__init__()
        self.implementation = Implementation.C

    def parse(self):
        return cde.HorizontalFlipOperation()


class HsvToRgb(PyTensorOperation):
    """
    Convert the input numpy.ndarray images from HSV to RGB.

    Args:
        is_hwc (bool): If True, means the input image is in shape of (H, W, C) or (N, H, W, C).
            Otherwise, it is in shape of (C, H, W) or (N, C, H, W). Default: False.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.transforms import Compose
        >>>
        >>> transforms_list = Compose([vision.Decode(to_pil=True)),
        ...                            vision.CenterCrop(20),
        ...                            vision.ToTensor(),
        ...                            vision.HsvToRgb()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_hsv_to_rgb
    def __init__(self, is_hwc=False):
        super().__init__()
        self.is_hwc = is_hwc
        self.random = False
        self.implementation = Implementation.PY

    def __call__(self, hsv_imgs):
        """
        Call method.

        Args:
            hsv_imgs (numpy.ndarray): HSV images to be converted.

        Returns:
            numpy.ndarray, converted RGB images.
        """
        return util.hsv_to_rgbs(hsv_imgs, self.is_hwc)


class HWC2CHW(TensorOperation, PyTensorOperation):
    """
    Transpose the input image from shape (H, W, C) to shape (C, H, W). The input image should be 3 channels image.

    Raises:
        RuntimeError: If given tensor shape is not <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [vision.Decode(),
        ...                    vision.RandomHorizontalFlip(0.75),
        ...                    vision.RandomCrop(512),
        ...                    vision.HWC2CHW()]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    def __init__(self):
        super().__init__()
        self.random = False

    def parse(self):
        return cde.HwcToChwOperation()

    def execute_pil(self, img):
        """
        Execute method.

        Args:
            img (numpy.ndarray): numpy.ndarray of shape (H, W, C) to be transposed.

        Returns:
            numpy.ndarray, transposed numpy.ndarray of shape (C, H, W).
        """
        return util.hwc_to_chw(img)


class Invert(TensorOperation, PyTensorOperation):
    """
    Apply invert on input image in RGB mode. This operator will reassign every pixel to (255 - pixel).

    Raises:
        RuntimeError: If given tensor shape is not <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [vision.Decode(), vision.Invert()]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    def __init__(self):
        super().__init__()
        self.random = False

    def parse(self):
        return cde.InvertOperation()

    def execute_pil(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be color inverted.

        Returns:
            PIL Image, color inverted image.
        """

        return util.invert_color(img)


class LinearTransformation(PyTensorOperation):
    r"""
    Linearly transform the input numpy.ndarray image with a square transformation matrix and a mean vector.

    It will first flatten the input image and subtract the mean vector from it, then compute the dot
    product with the transformation matrix, finally reshape it back to its original shape.

    Args:
        transformation_matrix (numpy.ndarray): A square transformation matrix in shape of (D, D), where
            :math:`D = C \times H \times W`.
        mean_vector (numpy.ndarray): A mean vector in shape of (D,), where :math:`D = C \times H \times W`.

    Raises:
        TypeError: If `transformation_matrix` is not of type :class:`numpy.ndarray`.
        TypeError: If `mean_vector` is not of type :class:`numpy.ndarray`.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore.dataset.transforms.transforms import Compose
        >>>
        >>> height, width = 32, 32
        >>> dim = 3 * height * width
        >>> transformation_matrix = np.ones([dim, dim])
        >>> mean_vector = np.zeros(dim)
        >>> transforms_list = Compose([vision.Decode(to_pil=True)),
        ...                            vision.Resize((height,width)),
        ...                            vision.ToTensor(),
        ...                            vision.LinearTransformation(transformation_matrix, mean_vector)])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_linear_transform
    def __init__(self, transformation_matrix, mean_vector):
        super().__init__()
        self.transformation_matrix = transformation_matrix
        self.mean_vector = mean_vector
        self.random = False
        self.implementation = Implementation.PY

    def __call__(self, np_img):
        """
        Call method.

        Args:
            np_img (numpy.ndarray): Image in shape of (C, H, W) to be linearly transformed.

        Returns:
            numpy.ndarray, linearly transformed image.
        """
        return util.linear_transform(np_img, self.transformation_matrix, self.mean_vector)


class MixUp(PyTensorOperation):
    """
    Randomly mix up a batch of images together with its labels.

    Each image will be multiplied by a random weight lambda generated from the Beta distribution and then added
    to another image multiplied by 1 - lambda. The same transformation will be applied to their labels with the
    same value of lambda. Make sure that the labels are one-hot encoded in advance.

    Args:
        batch_size (int): The number of images in a batch.
        alpha (float): The alpha and beta parameter for the Beta distribution.
        is_single (bool, optional): If True, it will randomly mix up [img0, ..., img(n-1), img(n)] with
            [img1, ..., img(n), img0] in each batch. Otherwise, it will randomly mix up images with the
            output of the previous batch. Default: True.

    Raises:
        TypeError: If `batch_size` is not of type integer.
        TypeError: If `alpha` is not of type float.
        TypeError: If `is_single` is not of type boolean.
        ValueError: If `batch_size` is not positive.
        ValueError: If `alpha` is not positive.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> # Setup multi-batch mixup transformation
        >>> transform = [vision.MixUp(batch_size=16, alpha=0.2, is_single=False)]
        >>> # Apply the transform to the dataset through dataset.map()
        >>> image_folder_dataset = image_folder_dataset.map(input_columns="image",
        ...                                                 operations=transform)
    """

    @check_mix_up
    def __init__(self, batch_size, alpha, is_single=True):
        super().__init__()
        self.image = 0
        self.label = 0
        self.is_first = True
        self.batch_size = batch_size
        self.alpha = alpha
        self.is_single = is_single
        self.random = False
        self.implementation = Implementation.PY

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


class MixUpBatch(TensorOperation):
    """
    Apply MixUp transformation on input batch of images and labels. Each image is
    multiplied by a random weight (lambda) and then added to a randomly selected image from the batch
    multiplied by (1 - lambda). The same formula is also applied to the one-hot labels.

    The lambda is generated based on the specified alpha value. Two coefficients x1, x2 are randomly generated
    in the range [alpha, 1], and lambda = (x1 / (x1 + x2)).

    Note that you need to make labels into one-hot format and batched before calling this operator.

    Args:
        alpha (float, optional): Hyperparameter of beta distribution. The value must be positive (default = 1.0).

    Raises:
        TypeError: If `alpha` is not of type float.
        ValueError: If `alpha` is not positive.
        RuntimeError: If given tensor shape is not <N, H, W, C> or <N, C, H, W>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> onehot_op = data_transforms.OneHot(num_classes=10)
        >>> image_folder_dataset= image_folder_dataset.map(operations=onehot_op,
        ...                                                input_columns=["label"])
        >>> mixup_batch_op = vision.MixUpBatch(alpha=0.9)
        >>> image_folder_dataset = image_folder_dataset.batch(5)
        >>> image_folder_dataset = image_folder_dataset.map(operations=mixup_batch_op,
        ...                                                 input_columns=["image", "label"])
    """

    @check_mix_up_batch_c
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.implementation = Implementation.C

    def parse(self):
        return cde.MixUpBatchOperation(self.alpha)


class Normalize(TensorOperation):
    """
    Normalize the input image with respect to mean and standard deviation. This operator will normalize
    the input image with: output[channel] = (input[channel] - mean[channel]) / std[channel], where channel >= 1.

    Args:
        mean (sequence): List or tuple of mean values for each channel, with respect to channel order.
            The mean values must be in range [0.0, 255.0].
        std (sequence): List or tuple of standard deviations for each channel, with respect to channel order.
            The standard deviation values must be in range (0.0, 255.0].
        is_hwc (bool): Whether the input image is in HWC format. True - HWC format, False - CHW format. Default: True.

    Raises:
        TypeError: If `mean` is not of type sequence.
        TypeError: If `std` is not of type sequence.
        ValueError: If `mean` is not in range [0.0, 255.0].
        ValueError: If `mean` is not in range (0.0, 255.0].

    Supported Platforms:
        ``CPU``

    Examples:
        >>> decode_op = vision.Decode() ## Decode output is expected to be HWC format
        >>> normalize_op = vision.Normalize(mean=[121.0, 115.0, 100.0], std=[70.0, 68.0, 71.0], is_hwc=True)
        >>> transforms_list = [decode_op, normalize_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_normalize
    def __init__(self, mean, std, is_hwc=True):
        super().__init__()
        self.mean = mean
        self.std = std
        self.is_hwc = is_hwc
        self.random = False
        self.implementation = Implementation.C

    def parse(self):
        return cde.NormalizeOperation(self.mean, self.std, self.is_hwc)


class NormalizePad(TensorOperation):
    """
    Normalize the input image with respect to mean and standard deviation then pad an extra channel with value zero.

    Args:
        mean (sequence): List or tuple of mean values for each channel, with respect to channel order.
            The mean values must be in range (0.0, 255.0].
        std (sequence): List or tuple of standard deviations for each channel, with respect to channel order.
            The standard deviation values must be in range (0.0, 255.0].
        dtype (str, optional): Set the output data type of normalized image (default is "float32").
        is_hwc (bool): Whether the input image is in HWC format. True - HWC format, False - CHW format. Default: True.

    Raises:
        TypeError: If `mean` is not of type sequence.
        TypeError: If `std` is not of type sequence.
        TypeError: If `dtype` is not of type string.
        ValueError: If `mean` is not in range [0.0, 255.0].
        ValueError: If `mean` is not in range (0.0, 255.0].

    Supported Platforms:
        ``CPU``

    Examples:
        >>> decode_op = vision.Decode()
        >>> normalize_pad_op = vision.NormalizePad(mean=[121.0, 115.0, 100.0],
        ...                                        std=[70.0, 68.0, 71.0],
        ...                                        dtype="float32")
        >>> transforms_list = [decode_op, normalize_pad_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_normalizepad
    def __init__(self, mean, std, dtype="float32", is_hwc=True):
        super().__init__()
        self.mean = mean
        self.std = std
        self.dtype = dtype
        self.is_hwc = is_hwc
        self.random = False
        self.implementation = Implementation.C

    def parse(self):
        return cde.NormalizePadOperation(self.mean, self.std, self.dtype, self.is_hwc)


class Pad(TensorOperation, PyTensorOperation):
    """
    Pad the image according to padding parameters.

    Args:
        padding (Union[int, Sequence[tuple]]): The number of pixels to pad each border of the image.
            If a single number is provided, it pads all borders with this value.
            If a tuple or lists of 2 values are provided, it pads the (left and top)
            with the first value and (right and bottom) with the second value.
            If 4 values are provided as a list or tuple, it pads the left, top, right and bottom respectively.
            The pad values must be non-negative.
        fill_value (Union[int, tuple[int]], optional): The pixel intensity of the borders, only valid for
            padding_mode Border.CONSTANT. If it is a 3-tuple, it is used to fill R, G, B channels respectively.
            If it is an integer, it is used for all RGB channels.
            The fill_value values must be in range [0, 255] (default=0).
        padding_mode (Border, optional): The method of padding (default=Border.CONSTANT). Can be any of
            [Border.CONSTANT, Border.EDGE, Border.REFLECT, Border.SYMMETRIC].

            - Border.CONSTANT, means it fills the border with constant values.

            - Border.EDGE, means it pads with the last value on the edge.

            - Border.REFLECT, means it reflects the values on the edge omitting the last
              value of edge.

            - Border.SYMMETRIC, means it reflects the values on the edge repeating the last
              value of edge.

    Raises:
        TypeError: If `padding` is not of type int or Sequence[int].
        TypeError: If `fill_value` is not of type int or tuple[int].
        TypeError: If `padding_mode` is not of type :class:`mindspore.dataset.vision.Border`.
        ValueError: If `padding` is negative.
        ValueError: If `fill_value` is not in range [0, 255].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [vision.Decode(), vision.Pad([100, 100, 100, 100])]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_pad
    def __init__(self, padding, fill_value=0, padding_mode=Border.CONSTANT):
        super().__init__()
        padding = parse_padding(padding)
        if isinstance(fill_value, int):
            fill_value = tuple([fill_value] * 3)
        self.padding = padding
        self.fill_value = fill_value
        self.padding_mode = padding_mode
        self.random = False
        self.c_padding_mode = Border.to_c_type(padding_mode)
        self.pil_padding_mode = Border.to_python_type(padding_mode)

    def parse(self):
        return cde.PadOperation(self.padding, self.fill_value, self.c_padding_mode)

    def execute_pil(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image, padded image.
        """
        return util.pad(img, self.padding, self.fill_value, self.pil_padding_mode)


class RandomAdjustSharpness(TensorOperation):
    """
    Randomly adjust the sharpness of the input image with a given probability.

    Args:
        degree (float): Sharpness adjustment degree, which must be non negative.
            Degree of 0.0 gives a blurred image, degree of 1.0 gives the original image,
            and degree of 2.0 increases the sharpness by a factor of 2.
        prob (float, optional): Probability of the image being sharpness adjusted, which
            must be in range of [0, 1] (default=0.5).

    Raises:
        TypeError: If `degree` is not of type float.
        TypeError: If `prob` is not of type float.
        ValueError: If `degree` is negative.
        ValueError: If `prob` is not in range [0, 1].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [vision.Decode(), vision.RandomAdjustSharpness(2.0, 0.5)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_random_adjust_sharpness
    def __init__(self, degree, prob=0.5):
        super().__init__()
        self.prob = prob
        self.degree = degree
        self.implementation = Implementation.C

    def parse(self):
        return cde.RandomAdjustSharpnessOperation(self.degree, self.prob)


class RandomAffine(TensorOperation, PyTensorOperation):
    """
    Apply Random affine transformation to the input image.

    Args:
        degrees (Union[int, float, sequence]): Range of the rotation degrees.
            If `degrees` is a number, the range will be (-degrees, degrees).
            If `degrees` is a sequence, it should be (min, max).
        translate (sequence, optional): Sequence (tx_min, tx_max, ty_min, ty_max) of minimum/maximum translation in
            x(horizontal) and y(vertical) directions, range [-1.0, 1.0] (default=None).
            The horizontal and vertical shift is selected randomly from the range:
            (tx_min*width, tx_max*width) and (ty_min*height, ty_max*height), respectively.
            If a tuple or list of size 2, then a translate parallel to the X axis in the range of
            (translate[0], translate[1]) is applied.
            If a tuple or list of size 4, then a translate parallel to the X axis in the range of
            (translate[0], translate[1]) and a translate parallel to the Y axis in the range of
            (translate[2], translate[3]) are applied.
            If None, no translation is applied.
        scale (sequence, optional): Scaling factor interval, which must be non negative
            (default=None, original scale is used).
        shear (Union[float, Sequence[float, float], Sequence[float, float, float, float]], optional):
            Range of shear factor to select from.
            If float is provided, a shearing parallel to X axis with a factor selected from
            (- `shear` , `shear` ) will be applied.
            If Sequence[float, float] is provided, a shearing parallel to X axis with a factor selected
            from ( `shear` [0], `shear` [1]) will be applied.
            If Sequence[float, float, float, float] is provided, a shearing parallel to X axis with a factor selected
            from ( `shear` [0], `shear` [1]) and a shearing parallel to Y axis with a factor selected from
            ( `shear` [2], `shear` [3]) will be applied. Default: None, means no shearing.
        resample (Inter, optional): An optional resampling filter (default=Inter.NEAREST).
            It can be any of [Inter.BILINEAR, Inter.NEAREST, Inter.BICUBIC, Inter.AREA].

            - Inter.BILINEAR, means resample method is bilinear interpolation.

            - Inter.NEAREST, means resample method is nearest-neighbor interpolation.

            - Inter.BICUBIC, means resample method is bicubic interpolation.

            - Inter.AREA, means resample method is pixel area interpolation.

        fill_value (Union[int, tuple[int]], optional): Optional fill_value to fill the area outside the transform
            in the output image. There must be three elements in tuple and the value of single element is [0, 255].
            (default=0, filling is performed).

    Raises:
        TypeError: If `degrees` is not of type int, float or sequence.
        TypeError: If `translate` is not of type sequence.
        TypeError: If `scale` is not of type sequence.
        TypeError: If `shear` is not of type int, float or sequence.
        TypeError: If `resample` is not of type :class:`mindspore.dataset.vision.Inter`.
        TypeError: If `fill_value` is not of type int or tuple[int].
        ValueError: If `degrees` is negative.
        ValueError: If `translate` is not in range [-1.0, 1.0].
        ValueError: If `scale` is negative.
        ValueError: If `shear` is not positive.
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.vision import Inter
        >>> decode_op = vision.Decode()
        >>> random_affine_op = vision.RandomAffine(degrees=15,
        ...                                        translate=(-0.1, 0.1, 0, 0),
        ...                                        scale=(0.9, 1.1),
        ...                                        resample=Inter.NEAREST)
        >>> transforms_list = [decode_op, random_affine_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_random_affine
    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=Inter.NEAREST, fill_value=0):
        super().__init__()
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
        self.scale = scale
        self.shear = shear
        self.c_resample = None
        self.py_resample = None
        if resample in [Inter.AREA, Inter.PILCUBIC]:
            self.c_resample = Inter.to_c_type(resample)
        elif resample == Inter.ANTIALIAS:
            self.py_resample = Inter.to_python_type(resample)
        else:
            self.c_resample = Inter.to_c_type(resample)
            self.py_resample = Inter.to_python_type(resample)
        self.fill_value = fill_value

    def parse(self):
        return cde.RandomAffineOperation(self.degrees, self.translate, self.scale, self.shear,
                                         self.c_resample, self.fill_value)

    def execute_pil(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be randomly affine transformed.

        Returns:
            PIL Image, randomly affine transformed image.
        """

        return util.random_affine(img,
                                  self.degrees,
                                  self.translate,
                                  self.scale,
                                  self.shear,
                                  self.py_resample,
                                  self.fill_value)


class RandomAutoContrast(TensorOperation):
    """
    Automatically adjust the contrast of the image with a given probability.

    Args:
        cutoff (float, optional): Percent of the lightest and darkest pixels to be cut off from
            the histogram of the input image. The value must be in range of [0.0, 50.0) (default=0.0).
        ignore (Union[int, sequence], optional): The background pixel values to be ignored, each of
            which must be in range of [0, 255] (default=None).
        prob (float, optional): Probability of the image being automatically contrasted, which
            must be in range of [0, 1] (default=0.5).

    Raises:
        TypeError: If `cutoff` is not of type float.
        TypeError: If `ignore` is not of type integer or sequence of integer.
        TypeError: If `prob` is not of type float.
        ValueError: If `cutoff` is not in range [0.0, 50.0).
        ValueError: If `ignore` is not in range [0, 255].
        ValueError: If `prob` is not in range [0, 1].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [vision.Decode(), vision.RandomAutoContrast(cutoff=0.0, ignore=None, prob=0.5)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_random_auto_contrast
    def __init__(self, cutoff=0.0, ignore=None, prob=0.5):
        super().__init__()
        if ignore is None:
            ignore = []
        if isinstance(ignore, int):
            ignore = [ignore]
        self.cutoff = cutoff
        self.ignore = ignore
        self.prob = prob
        self.implementation = Implementation.C

    def parse(self):
        return cde.RandomAutoContrastOperation(self.cutoff, self.ignore, self.prob)


class RandomColor(TensorOperation, PyTensorOperation):
    """
    Adjust the color of the input image by a fixed or random degree.
    This operation works only with 3-channel color images.

    Args:
         degrees (Sequence[float], optional): Range of random color adjustment degrees, which must be non-negative.
            It should be in (min, max) format. If min=max, then it is a
            single fixed magnitude operation (default=(0.1, 1.9)).

    Raises:
        TypeError: If `degrees` is not of type Sequence[float].
        ValueError: If `degrees` is negative.
        RuntimeError: If given tensor shape is not <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [vision.Decode(), vision.RandomColor((0.5, 2.0))]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_positive_degrees
    def __init__(self, degrees=(0.1, 1.9)):
        super().__init__()
        self.degrees = degrees

    def parse(self):
        return cde.RandomColorOperation(*self.degrees)

    def execute_pil(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be color adjusted.

        Returns:
            PIL Image, color adjusted image.
        """

        return util.random_color(img, self.degrees)


class RandomColorAdjust(TensorOperation, PyTensorOperation):
    """
    Randomly adjust the brightness, contrast, saturation, and hue of the input image.

    Args:
        brightness (Union[float, Sequence[float]], optional): Brightness adjustment factor (default=(1, 1)).
            Cannot be negative.
            If it is a float, the factor is uniformly chosen from the range [max(0, 1-brightness), 1+brightness].
            If it is a sequence, it should be [min, max] for the range.
        contrast (Union[float, Sequence[float]], optional): Contrast adjustment factor (default=(1, 1)).
            Cannot be negative.
            If it is a float, the factor is uniformly chosen from the range [max(0, 1-contrast), 1+contrast].
            If it is a sequence, it should be [min, max] for the range.
        saturation (Union[float, Sequence[float]], optional): Saturation adjustment factor (default=(1, 1)).
            Cannot be negative.
            If it is a float, the factor is uniformly chosen from the range [max(0, 1-saturation), 1+saturation].
            If it is a sequence, it should be [min, max] for the range.
        hue (Union[float, Sequence[float]], optional): Hue adjustment factor (default=(0, 0)).
            If it is a float, the range will be [-hue, hue]. Value should be 0 <= hue <= 0.5.
            If it is a sequence, it should be [min, max] where -0.5 <= min <= max <= 0.5.

    Raises:
        TypeError: If `brightness` is not of type float or Sequence[float].
        TypeError: If `contrast` is not of type float or Sequence[float].
        TypeError: If `saturation` is not of type float or Sequence[float].
        TypeError: If `hue` is not of type float or Sequence[float].
        ValueError: If `brightness` is negative.
        ValueError: If `contrast` is negative.
        ValueError: If `saturation` is negative.
        ValueError: If `hue` is not in range [-0.5, 0.5].
        RuntimeError: If given tensor shape is not <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> decode_op = vision.Decode()
        >>> transform_op = vision.RandomColorAdjust(brightness=(0.5, 1),
        ...                                         contrast=(0.4, 1),
        ...                                         saturation=(0.3, 1))
        >>> transforms_list = [decode_op, transform_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_random_color_adjust
    def __init__(self, brightness=(1, 1), contrast=(1, 1), saturation=(1, 1), hue=(0, 0)):
        super().__init__()
        brightness = self.__expand_values(brightness)
        contrast = self.__expand_values(contrast)
        saturation = self.__expand_values(saturation)
        hue = self.__expand_values(
            hue, center=0, bound=(-0.5, 0.5), non_negative=False)

        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def parse(self):
        return cde.RandomColorAdjustOperation(self.brightness, self.contrast, self.saturation, self.hue)

    def execute_pil(self, img):
        """
        Execute method.

        Args:
            img (PIL image): Image to be randomly color adjusted.

        Returns:
            PIL Image, randomly color adjusted image.
        """
        return util.random_color_adjust(img, self.brightness, self.contrast, self.saturation, self.hue)

    def __expand_values(self, value, center=1, bound=(0, FLOAT_MAX_INTEGER), non_negative=True):
        """Expand input value for vision adjustment factor."""
        if isinstance(value, numbers.Number):
            value = [center - value, center + value]
            if non_negative:
                value[0] = max(0, value[0])
            check_range(value, bound)
        return (value[0], value[1])


class RandomCrop(TensorOperation, PyTensorOperation):
    """
    Crop the input image at a random location. If input image size is smaller than output size,
    input image will be padded before cropping.

    Note:
        If the input image is more than one, then make sure that the image size is the same.

    Args:
        size (Union[int, Sequence[int]]): The output size of the cropped image. The size value(s) must be positive.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, an image of size (height, width) will be cropped.
        padding (Union[int, Sequence[int]], optional): The number of pixels to pad each border of the image.
            The padding value(s) must be non-nagetive (default=None).
            If padding is not None, pad image first with padding values.
            If a single number is provided, pad all borders with this value.
            If a tuple or lists of 2 values are provided, pad the (left and top)
            with the first value and (right and bottom) with the second value.
            If 4 values are provided as a list or tuple,
            pad the left, top, right and bottom respectively.
        pad_if_needed (bool, optional): Pad the image if either side is smaller than
            the given output size (default=False).
        fill_value (Union[int, tuple[int]], optional): The pixel intensity of the borders, only valid for
            padding_mode Border.CONSTANT. If it is a 3-tuple, it is used to fill R, G, B channels respectively.
            If it is an integer, it is used for all RGB channels.
            The fill_value values must be in range [0, 255] (default=0).
        padding_mode (Border, optional): The method of padding (default=Border.CONSTANT). It can be any of
            [Border.CONSTANT, Border.EDGE, Border.REFLECT, Border.SYMMETRIC].

            - Border.CONSTANT, means it fills the border with constant values.

            - Border.EDGE, means it pads with the last value on the edge.

            - Border.REFLECT, means it reflects the values on the edge omitting the last
              value of edge.

            - Border.SYMMETRIC, means it reflects the values on the edge repeating the last
              value of edge.

    Raises:
        TypeError: If `size` is not of type int or Sequence[int].
        TypeError: If `padding` is not of type int or Sequence[int].
        TypeError: If `pad_if_needed` is not of type boolean.
        TypeError: If `fill_value` is not of type int or tuple[int].
        TypeError: If `padding_mode` is not of type :class:`mindspore.dataset.vision.Border`.
        ValueError: If `size` is not positive.
        ValueError: If `padding` is negative.
        ValueError: If `fill_value` is not in range [0, 255].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.vision import Border
        >>> decode_op = vision.Decode()
        >>> random_crop_op = vision.RandomCrop(512, [200, 200, 200, 200], padding_mode=Border.EDGE)
        >>> transforms_list = [decode_op, random_crop_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_random_crop
    def __init__(self, size, padding=None, pad_if_needed=False, fill_value=0, padding_mode=Border.CONSTANT):
        super().__init__()
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
        self.c_padding_mode = Border.to_c_type(padding_mode)
        self.pil_padding_mode = Border.to_python_type(padding_mode)

    def parse(self):
        return cde.RandomCropOperation(self.size, self.padding, self.pad_if_needed, self.fill_value,
                                       self.c_padding_mode)

    def execute_pil(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be randomly cropped.

        Returns:
            PIL Image, cropped image.
        """
        return util.random_crop(img, self.size, self.padding, self.pad_if_needed,
                                self.fill_value, self.pil_padding_mode)


class RandomCropDecodeResize(TensorOperation):
    """
    A combination of `Crop`, `Decode` and `Resize`. It will get better performance for JPEG images. This operator
    will crop the input image at a random location, decode the cropped image in RGB mode, and resize the decoded image.

    Args:
        size (Union[int, Sequence[int]]): The output size of the resized image. The size value(s) must be positive.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).
        scale (list, tuple, optional): Range [min, max) of respective size of the
            original size to be cropped, which must be non-negative (default=(0.08, 1.0)).
        ratio (list, tuple, optional): Range [min, max) of aspect ratio to be
            cropped, which must be non-negative (default=(3. / 4., 4. / 3.)).
        interpolation (Inter, optional): Image interpolation mode for resize operator(default=Inter.BILINEAR).
            It can be any of [Inter.BILINEAR, Inter.NEAREST, Inter.BICUBIC, Inter.AREA, Inter.PILCUBIC].

            - Inter.BILINEAR, means interpolation method is bilinear interpolation.

            - Inter.NEAREST, means interpolation method is nearest-neighbor interpolation.

            - Inter.BICUBIC, means interpolation method is bicubic interpolation.

            - Inter.AREA, means interpolation method is pixel area interpolation.

            - Inter.PILCUBIC, means interpolation method is bicubic interpolation like implemented in pillow, input
              should be in 3 channels format.

        max_attempts (int, optional): The maximum number of attempts to propose a valid crop_area (default=10).
            If exceeded, fall back to use center_crop instead. The max_attempts value must be positive.

    Raises:
        TypeError: If `size` is not of type int or Sequence[int].
        TypeError: If `scale` is not of type tuple.
        TypeError: If `ratio` is not of type tuple.
        TypeError: If `interpolation` is not of type :class:`mindspore.dataset.vision.Inter`.
        TypeError: If `max_attempts` is not of type integer.
        ValueError: If `size` is not positive.
        ValueError: If `scale` is negative.
        ValueError: If `ratio` is negative.
        ValueError: If `max_attempts` is not positive.
        RuntimeError: If given tensor is not a 1D sequence.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.vision import Inter
        >>> resize_crop_decode_op = vision.RandomCropDecodeResize(size=(50, 75),
        ...                                                       scale=(0.25, 0.5),
        ...                                                       interpolation=Inter.NEAREST,
        ...                                                       max_attempts=5)
        >>> transforms_list = [resize_crop_decode_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_random_resize_crop
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation=Inter.BILINEAR, max_attempts=10):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.max_attempts = max_attempts
        self.implementation = Implementation.C

    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            raise TypeError(
                "Input should be an encoded image in 1-D NumPy format, got {}.".format(type(img)))
        if img.ndim != 1 or img.dtype.type is not np.uint8:
            raise TypeError("Input should be an encoded image with uint8 type in 1-D NumPy format, " +
                            "got format:{}, dtype:{}.".format(type(img), img.dtype.type))
        return super().__call__(img)

    def parse(self):
        return cde.RandomCropDecodeResizeOperation(self.size, self.scale, self.ratio,
                                                   Inter.to_c_type(self.interpolation),
                                                   self.max_attempts)


class RandomCropWithBBox(TensorOperation):
    """
    Crop the input image at a random location and adjust bounding boxes accordingly.

    Args:
        size (Union[int, Sequence[int]]): The output size of the cropped image. The size value(s) must be positive.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, an image of size (height, width) will be cropped.
        padding (Union[int, Sequence[int]], optional): The number of pixels to pad the image
            The padding value(s) must be non-nagetive (default=None).
            If padding is not None, first pad image with padding values.
            If a single number is provided, pad all borders with this value.
            If a tuple or lists of 2 values are provided, pad the (left and top)
            with the first value and (right and bottom) with the second value.
            If 4 values are provided as a list or tuple, pad the left, top, right and bottom respectively.
        pad_if_needed (bool, optional): Pad the image if either side is smaller than
            the given output size (default=False).
        fill_value (Union[int, tuple[int]], optional): The pixel intensity of the borders, only valid for
            padding_mode Border.CONSTANT. If it is a 3-tuple, it is used to fill R, G, B channels respectively.
            If it is an integer, it is used for all RGB channels.
            The fill_value values must be in range [0, 255] (default=0).
        padding_mode (Border, optional): The method of padding (default=Border.CONSTANT). It can be any of
            [Border.CONSTANT, Border.EDGE, Border.REFLECT, Border.SYMMETRIC].

            - Border.CONSTANT, means it fills the border with constant values.

            - Border.EDGE, means it pads with the last value on the edge.

            - Border.REFLECT, means it reflects the values on the edge omitting the last
              value of edge.

            - Border.SYMMETRIC, means it reflects the values on the edge repeating the last

              value of edge.

    Raises:
        TypeError: If `size` is not of type int or Sequence[int].
        TypeError: If `padding` is not of type int or Sequence[int].
        TypeError: If `pad_if_needed` is not of type boolean.
        TypeError: If `fill_value` is not of type int or tuple[int].
        TypeError: If `padding_mode` is not of type :class:`mindspore.dataset.vision.Border`.
        ValueError: If `size` is not positive.
        ValueError: If `padding` is negative.
        ValueError: If `fill_value` is not in range [0, 255].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> decode_op = vision.Decode()
        >>> random_crop_with_bbox_op = vision.RandomCropWithBBox([512, 512], [200, 200, 200, 200])
        >>> transforms_list = [decode_op, random_crop_with_bbox_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_random_crop
    def __init__(self, size, padding=None, pad_if_needed=False, fill_value=0, padding_mode=Border.CONSTANT):
        super().__init__()
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
        self.implementation = Implementation.C

    def parse(self):
        border_type = Border.to_c_type(self.padding_mode)
        return cde.RandomCropWithBBoxOperation(self.size, self.padding, self.pad_if_needed, self.fill_value,
                                               border_type)


class RandomEqualize(TensorOperation):
    """
    Apply histogram equalization on the input image with a given probability.

    Args:
        prob (float, optional): Probability of the image being equalized, which
            must be in range of [0, 1] (default=0.5).

    Raises:
        TypeError: If `prob` is not of type float.
        ValueError: If `prob` is not in range [0, 1].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [vision.Decode(), vision.RandomEqualize(0.5)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_prob
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob
        self.implementation = Implementation.C

    def parse(self):
        return cde.RandomEqualizeOperation(self.prob)


class RandomErasing(PyTensorOperation):
    """
    Randomly erase pixels within a random selected rectangle erea on the input numpy.ndarray image.

    See `Random Erasing Data Augmentation <https://arxiv.org/pdf/1708.04896.pdf>`_.

    Args:
        prob (float, optional): Probability of performing erasing. Default: 0.5.
        scale (sequence, optional): Range of area scale of the erased area relative
            to the original image to select from, arranged in order of (min, max).
            Default: (0.02, 0.33).
        ratio (sequence, optional): Range of aspect ratio of the erased area to select
            from, arraged in order of (min, max). Default: (0.3, 3.3).
        value (Union[int, str, sequence]): Pixel value used to pad the erased area.
            If a single integer is provided, it will be used for all RGB channels.
            If a sequence of length 3 is provided, it will be used for R, G, B channels respectively.
            If a string of 'random' is provided, each pixel will be erased with a random value obtained
            from a standard normal distribution. Default: 0.
        inplace (bool, optional): Whether to apply erasing inplace. Default: False.
        max_attempts (int, optional): The maximum number of attempts to propose a valid
            erased area, beyond which the original image will be returned. Default: 10.

    Raises:
        TypeError: If `prob` is not of type float.
        TypeError: If `scale` is not of type sequence.
        TypeError: If `ratio` is not of type sequence.
        TypeError: If `value` is not of type integer, string, or sequence.
        TypeError: If `inplace` is not of type boolean.
        TypeError: If `max_attempts` is not of type integer.
        ValueError: If `prob` is not in range of [0, 1].
        ValueError: If `scale` is negative.
        ValueError: If `ratio` is negative.
        ValueError: If `value` is not in range of [0, 255].
        ValueError: If `max_attempts` is not positive.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.transforms import Compose
        >>>
        >>> transforms_list = Compose([vision.Decode(to_pil=True)),
        ...                            vision.ToTensor(),
        ...                            vision.RandomErasing(value='random')])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_random_erasing
    def __init__(self, prob=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False, max_attempts=10):
        super().__init__()
        self.prob = prob
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace
        self.max_attempts = max_attempts
        self.implementation = Implementation.PY

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


class RandomGrayscale(PyTensorOperation):
    """
    Randomly convert the input PIL Image to grayscale.

    Args:
        prob (float, optional): Probability of performing grayscale conversion. Default: 0.1.

    Raises:
        TypeError: If `prob` is not of type float.
        ValueError: If `prob` is not in range of [0, 1].

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.transforms import Compose
        >>>
        >>> transforms_list = Compose([vision.Decode(to_pil=True)),
        ...                            vision.RandomGrayscale(0.3),
        ...                            vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_prob
    def __init__(self, prob=0.1):
        super().__init__()
        self.prob = prob
        self.implementation = Implementation.PY

    def execute_pil(self, img):
        """
        Execute method.

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


class RandomHorizontalFlip(TensorOperation, PyTensorOperation):
    """
    Randomly flip the input image horizontally with a given probability.

    Args:
        prob (float, optional): Probability of the image being flipped, which must be in range of [0, 1] (default=0.5).

    Raises:
        TypeError: If `prob` is not of type float.
        ValueError: If `prob` is not in range [0, 1].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [vision.Decode(), vision.RandomHorizontalFlip(0.75)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_prob
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def parse(self):
        return cde.RandomHorizontalFlipOperation(self.prob)

    def execute_pil(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be horizontally flipped.

        Returns:
            PIL Image, randomly horizontally flipped image.
        """
        return util.random_horizontal_flip(img, self.prob)


class RandomHorizontalFlipWithBBox(TensorOperation):
    """
    Flip the input image horizontally randomly with a given probability and adjust bounding boxes accordingly.

    Args:
        prob (float, optional): Probability of the image being flipped, which must be in range of [0, 1] (default=0.5).

    Raises:
        TypeError: If `prob` is not of type float.
        ValueError: If `prob` is not in range [0, 1].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [vision.Decode(), vision.RandomHorizontalFlipWithBBox(0.70)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_prob
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob
        self.implementation = Implementation.C

    def parse(self):
        return cde.RandomHorizontalFlipWithBBoxOperation(self.prob)


class RandomInvert(TensorOperation):
    """
    Randomly invert the colors of image with a given probability.

    Args:
        prob (float, optional): Probability of the image being inverted, which must be in range of [0, 1] (default=0.5).

    Raises:
        TypeError: If `prob` is not of type float.
        ValueError: If `prob` is not in range [0, 1].
        RuntimeError: If given tensor shape is not <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [vision.Decode(), vision.RandomInvert(0.5)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_prob
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob
        self.implementation = Implementation.C

    def parse(self):
        return cde.RandomInvertOperation(self.prob)


class RandomLighting(TensorOperation, PyTensorOperation):
    """
    Add AlexNet-style PCA-based noise to an image. The eigenvalue and eigenvectors for Alexnet's PCA noise is
    calculated from the imagenet dataset.

    Args:
        alpha (float, optional): Intensity of the image, which must be non-negative (default=0.05).

    Raises:
        TypeError: If `alpha` is not of type float.
        ValueError: If `alpha` is negative.
        RuntimeError: If given tensor shape is not <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [vision.Decode(), vision.RandomLighting(0.1)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_alpha
    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha

    def parse(self):
        return cde.RandomLightingOperation(self.alpha)

    def execute_pil(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be added AlexNet-style PCA-based noise.

        Returns:
            PIL Image, image with noise added.
        """

        return util.random_lighting(img, self.alpha)


class RandomPerspective(PyTensorOperation):
    """
    Randomly apply perspective transformation to the input PIL Image with a given probability.

    Args:
        distortion_scale (float, optional): Scale of distortion, in range of [0, 1]. Default: 0.5.
        prob (float, optional): Probability of performing perspective transformation. Default: 0.5.
        interpolation (Inter, optional): Method of interpolation. It can be Inter.BILINEAR,
            Inter.NEAREST or Inter.BICUBIC. Default: Inter.BICUBIC.

            - Inter.BILINEAR, bilinear interpolation.
            - Inter.NEAREST, nearest-neighbor interpolation.
            - Inter.BICUBIC, bicubic interpolation.

    Raises:
        TypeError: If `distortion_scale` is not of type float.
        TypeError: If `prob` is not of type float.
        TypeError: If `interpolation` is not of type :class:`mindspore.dataset.vision.Inter`.
        ValueError: If `distortion_scale` is not in range of [0, 1].
        ValueError: If `prob` is not in range of [0, 1].

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.transforms import Compose
        >>>
        >>> transforms_list = Compose([vision.Decode(to_pil=True)),
        ...                            vision.RandomPerspective(prob=0.1),
        ...                            vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_random_perspective
    def __init__(self, distortion_scale=0.5, prob=0.5, interpolation=Inter.BICUBIC):
        super().__init__()
        self.distortion_scale = distortion_scale
        self.prob = prob
        self.interpolation = Inter.to_python_type(interpolation)
        self.implementation = Implementation.PY

    def execute_pil(self, img):
        """
        Execute method.

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


class RandomPosterize(TensorOperation):
    """
    Reduce the number of bits for each color channel to posterize the input image randomly with a given probability.

    Args:
        bits (sequence or int, optional): Range of random posterize to compress image.
            Bits values must be in range of [1,8], and include at
            least one integer value in the given range. It must be in
            (min, max) or integer format. If min=max, then it is a single fixed
            magnitude operation (default=(8, 8)).

    Raises:
        TypeError: If `bits` is not of type integer or sequence of integer.
        ValueError: If `bits` is not in range [1, 8].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [vision.Decode(), vision.RandomPosterize((6, 8))]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_posterize
    def __init__(self, bits=(8, 8)):
        super().__init__()
        self.bits = bits
        self.implementation = Implementation.C

    def parse(self):
        bits = self.bits
        if isinstance(bits, int):
            bits = (bits, bits)
        return cde.RandomPosterizeOperation(bits)


class RandomResizedCrop(TensorOperation, PyTensorOperation):
    """
    This operator will crop the input image randomly, and resize the cropped image using a selected interpolation mode.

    Note:
        If the input image is more than one, then make sure that the image size is the same.

    Args:
        size (Union[int, Sequence[int]]): The output size of the resized image. The size value(s) must be positive.
            If size is an integer, a square of size (size, size) will be cropped with this value.
            If size is a sequence of length 2, an image of size (height, width) will be cropped.
        scale (Union[list, tuple], optional): Range [min, max) of respective size of the original
            size to be cropped, which must be non-negative (default=(0.08, 1.0)).
        ratio (Union[list, tuple], optional): Range [min, max) of aspect ratio to be
            cropped, which must be non-negative (default=(3. / 4., 4. / 3.)).
        interpolation (Inter, optional): Method of interpolation (default=Inter.BILINEAR).
            It can be any of [Inter.BILINEAR, Inter.NEAREST, Inter.BICUBIC, Inter.AREA, Inter.PILCUBIC].

            - Inter.BILINEAR, means interpolation method is bilinear interpolation.

            - Inter.NEAREST, means interpolation method is nearest-neighbor interpolation.

            - Inter.BICUBIC, means interpolation method is bicubic interpolation.

            - Inter.AREA, means interpolation method is pixel area interpolation.

            - Inter.PILCUBIC, means interpolation method is bicubic interpolation like implemented in pillow, input
              should be in 3 channels format.

        max_attempts (int, optional): The maximum number of attempts to propose a valid
            crop_area (default=10). If exceeded, fall back to use center_crop instead.

    Raises:
        TypeError: If `size` is not of type int or Sequence[int].
        TypeError: If `scale` is not of type tuple or list.
        TypeError: If `ratio` is not of type tuple or list.
        TypeError: If `interpolation` is not of type :class:`mindspore.dataset.vision.Inter`.
        TypeError: If `max_attempts` is not of type int.
        ValueError: If `size` is not positive.
        ValueError: If `scale` is negative.
        ValueError: If `ratio` is negative.
        ValueError: If `max_attempts` is not positive.
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.vision import Inter
        >>> decode_op = vision.Decode()
        >>> resize_crop_op = vision.RandomResizedCrop(size=(50, 75), scale=(0.25, 0.5),
        ...                                           interpolation=Inter.BILINEAR)
        >>> transforms_list = [decode_op, resize_crop_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_random_resize_crop
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation=Inter.BILINEAR, max_attempts=10):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.c_interpolation = None
        self.py_interpolation = None
        if interpolation in [Inter.AREA, Inter.PILCUBIC]:
            self.c_interpolation = Inter.to_c_type(interpolation)
        elif interpolation == Inter.ANTIALIAS:
            self.py_interpolation = Inter.to_python_type(interpolation)
        else:
            self.c_interpolation = Inter.to_c_type(interpolation)
            self.py_interpolation = Inter.to_python_type(interpolation)
        self.max_attempts = max_attempts

    def parse(self):
        return cde.RandomResizedCropOperation(self.size, self.scale, self.ratio, self.c_interpolation,
                                              self.max_attempts)

    def execute_pil(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be randomly cropped and resized.

        Returns:
            PIL Image, randomly cropped and resized image.
        """
        return util.random_resize_crop(img, self.size, self.scale, self.ratio,
                                       self.py_interpolation, self.max_attempts)


class RandomResizedCropWithBBox(TensorOperation):
    """
    Crop the input image to a random size and aspect ratio and adjust bounding boxes accordingly.

    Args:
        size (Union[int, Sequence[int]]): The size of the output image. The size value(s) must be positive.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).
        scale (list, tuple, optional): Range (min, max) of respective size of the original
            size to be cropped, which must be non-negative (default=(0.08, 1.0)).
        ratio (list, tuple, optional): Range (min, max) of aspect ratio to be
            cropped, which must be non-negative (default=(3. / 4., 4. / 3.)).
        interpolation (Inter mode, optional): Image interpolation mode (default=Inter.BILINEAR).
            It can be any of [Inter.BILINEAR, Inter.NEAREST, Inter.BICUBIC].

            - Inter.BILINEAR, means interpolation method is bilinear interpolation.

            - Inter.NEAREST, means interpolation method is nearest-neighbor interpolation.

            - Inter.BICUBIC, means interpolation method is bicubic interpolation.

        max_attempts (int, optional): The maximum number of attempts to propose a valid
            crop area (default=10). If exceeded, fall back to use center crop instead.

    Raises:
        TypeError: If `size` is not of type int or Sequence[int].
        TypeError: If `scale` is not of type tuple.
        TypeError: If `ratio` is not of type tuple.
        TypeError: If `interpolation` is not of type Inter.
        TypeError: If `max_attempts` is not of type integer.
        ValueError: If `size` is not positive.
        ValueError: If `scale` is negative.
        ValueError: If `ratio` is negative.
        ValueError: If `max_attempts` is not positive.
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.vision import Inter
        >>> decode_op = vision.Decode()
        >>> bbox_op = vision.RandomResizedCropWithBBox(size=50, interpolation=Inter.NEAREST)
        >>> transforms_list = [decode_op, bbox_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_random_resize_crop
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation=Inter.BILINEAR, max_attempts=10):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.max_attempts = max_attempts
        self.implementation = Implementation.C

    def parse(self):
        return cde.RandomResizedCropWithBBoxOperation(self.size, self.scale, self.ratio,
                                                      Inter.to_c_type(self.interpolation), self.max_attempts)


class RandomResize(TensorOperation):
    """
    Resize the input image using a randomly selected interpolation mode.

    Args:
        size (Union[int, Sequence[int]]): The output size of the resized image. The size value(s) must be positive.
            If size is an integer, smaller edge of the image will be resized to this value with
            the same image aspect ratio.
            If size is a sequence of length 2, it should be (height, width).

    Raises:
        TypeError: If `size` is not of type int or Sequence[int].
        ValueError: If `size` is not positive.
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> # randomly resize image, keeping aspect ratio
        >>> transforms_list1 = [vision.Decode(), vision.RandomResize(50)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list1,
        ...                                                 input_columns=["image"])
        >>> # randomly resize image to landscape style
        >>> transforms_list2 = [vision.Decode(), vision.RandomResize((40, 60))]
        >>> image_folder_dataset_1 = image_folder_dataset_1.map(operations=transforms_list2,
        ...                                                     input_columns=["image"])
    """

    @check_resize
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.implementation = Implementation.C

    def parse(self):
        size = self.size
        if isinstance(size, int):
            size = (size,)
        return cde.RandomResizeOperation(size)


class RandomResizeWithBBox(TensorOperation):
    """
    Tensor operation to resize the input image using a randomly selected interpolation mode and adjust
    bounding boxes accordingly.

    Args:
        size (Union[int, Sequence[int]]): The output size of the resized image. The size value(s) must be positive.
            If size is an integer, smaller edge of the image will be resized to this value with
            the same image aspect ratio.
            If size is a sequence of length 2, it should be (height, width).

    Raises:
        TypeError: If `size` is not of type int or Sequence[int].
        ValueError: If `size` is not positive.
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> # randomly resize image with bounding boxes, keeping aspect ratio
        >>> transforms_list1 = [vision.Decode(), vision.RandomResizeWithBBox(60)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list1,
        ...                                                 input_columns=["image"])
        >>> # randomly resize image with bounding boxes to portrait style
        >>> transforms_list2 = [vision.Decode(), vision.RandomResizeWithBBox((80, 60))]
        >>> image_folder_dataset_1 = image_folder_dataset_1.map(operations=transforms_list2,
        ...                                                     input_columns=["image"])
    """

    @check_resize
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.implementation = Implementation.C

    def parse(self):
        size = self.size
        if isinstance(size, int):
            size = (size,)
        return cde.RandomResizeWithBBoxOperation(size)


class RandomRotation(TensorOperation, PyTensorOperation):
    """
    Rotate the input image randomly within a specified range of degrees.

    Args:
        degrees (Union[int, float, sequence]): Range of random rotation degrees.
            If `degrees` is a number, the range will be converted to (-degrees, degrees).
            If `degrees` is a sequence, it should be (min, max).
        resample (Inter, optional): An optional resampling filter (default=Inter.NEAREST).
            It can be any of [Inter.BILINEAR, Inter.NEAREST, Inter.BICUBIC, Inter.AREA, Inter.ANTIALIAS].

            - Inter.BILINEAR, means resample method is bilinear interpolation.

            - Inter.NEAREST, means resample method is nearest-neighbor interpolation.

            - Inter.BICUBIC, means resample method is bicubic interpolation.

            - Inter.AREA, means the interpolation method is area interpolation.

            - Inter.ANTIALIAS, means the interpolation method is antialias interpolation.

        expand (bool, optional):  Optional expansion flag (default=False). If set to True, expand the output
            image to make it large enough to hold the entire rotated image.
            If set to False or omitted, make the output image the same size as the input.
            Note that the expand flag assumes rotation around the center and no translation.
        center (tuple, optional): Optional center of rotation (a 2-tuple) (default=None).
            Origin is the top left corner. None sets to the center of the image.
        fill_value (Union[int, tuple[int]], optional): Optional fill color for the area outside the rotated image.
            If it is a 3-tuple, it is used to fill R, G, B channels respectively.
            If it is an integer, it is used for all RGB channels.
            The fill_value values must be in range [0, 255] (default=0).

    Raises:
        TypeError: If `degrees` is not of type integer, float or sequence.
        TypeError: If `resample` is not of type Inter.
        TypeError: If `expand` is not of type boolean.
        TypeError: If `center` is not of type tuple.
        TypeError: If `fill_value` is not of type int or tuple[int].
        ValueError: If `fill_value` is not in range [0, 255].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.vision import Inter
        >>> transforms_list = [vision.Decode(),
        ...                    vision.RandomRotation(degrees=5.0,
        ...                    resample=Inter.NEAREST,
        ...                    expand=True)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_random_rotation
    def __init__(self, degrees, resample=Inter.NEAREST, expand=False, center=None, fill_value=0):
        super().__init__()
        if isinstance(degrees, (int, float)):
            degrees = degrees % 360
            degrees = [-degrees, degrees]
        elif isinstance(degrees, (list, tuple)):
            if degrees[1] - degrees[0] >= 360:
                degrees = [-180, 180]
            else:
                degrees = [degrees[0] % 360, degrees[1] % 360]
                if degrees[0] > degrees[1]:
                    degrees[1] += 360
        if isinstance(fill_value, int):
            fill_value = tuple([fill_value] * 3)
        self.degrees = degrees
        self.c_resample = None
        self.py_resample = None
        if resample in [Inter.AREA, Inter.PILCUBIC]:
            self.c_resample = Inter.to_c_type(resample)
        elif resample == Inter.ANTIALIAS:
            self.py_resample = Inter.to_python_type(resample)
        else:
            self.c_resample = Inter.to_c_type(resample)
            self.py_resample = Inter.to_python_type(resample)
        self.expand = expand
        self.py_center = center
        self.c_center = center
        if center is None:
            self.c_center = ()
        self.fill_value = fill_value

    def parse(self):
        return cde.RandomRotationOperation(self.degrees, self.c_resample, self.expand, self.c_center,
                                           self.fill_value)

    def execute_pil(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be randomly rotated.

        Returns:
            PIL Image, randomly rotated image.
        """
        return util.random_rotation(img, self.degrees, self.py_resample, self.expand, self.py_center, self.fill_value)


class RandomSelectSubpolicy(TensorOperation):
    """
    Choose a random sub-policy from a policy list to be applied on the input image.

    Args:
        policy (list[list[tuple[TensorOperation, float]]]): List of sub-policies to choose from.
            A sub-policy is a list of tuple[operation, prob], where operation is a data processing operation and prob
            is the probability that this operation will be applied, and the prob values must be in range [0, 1].
            Once a sub-policy is selected, each operation within the sub-policy with be applied in sequence according
            to its probability.

    Raises:
        TypeError: If `policy` contains invalid data processing operations.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> policy = [[(vision.RandomRotation((45, 45)), 0.5),
        ...            (vision.RandomVerticalFlip(), 1),
        ...            (vision.RandomColorAdjust(), 0.8)],
        ...           [(vision.RandomRotation((90, 90)), 1),
        ...            (vision.RandomColorAdjust(), 0.2)]]
        >>> image_folder_dataset = image_folder_dataset.map(operations=vision.RandomSelectSubpolicy(policy),
        ...                                                 input_columns=["image"])
    """

    @check_random_select_subpolicy_op
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.implementation = Implementation.C

    def parse(self):
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


class RandomSharpness(TensorOperation, PyTensorOperation):
    """
    Adjust the sharpness of the input image by a fixed or random degree. Degree of 0.0 gives a blurred image,
    degree of 1.0 gives the original image, and degree of 2.0 gives a sharpened image.

    Args:
        degrees (Union[list, tuple], optional): Range of random sharpness adjustment degrees,
            which must be non-negative. It should be in (min, max) format. If min=max, then
            it is a single fixed magnitude operation (default = (0.1, 1.9)).

    Raises:
        TypeError : If `degrees` is not a list or a tuple.
        ValueError: If `degrees` is negative.
        ValueError: If `degrees` is in (max, min) format instead of (min, max).

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [vision.Decode(), vision.RandomSharpness(degrees=(0.2, 1.9))]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_positive_degrees
    def __init__(self, degrees=(0.1, 1.9)):
        super().__init__()
        self.degrees = degrees

    def parse(self):
        return cde.RandomSharpnessOperation(self.degrees)

    def execute_pil(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be sharpness adjusted.

        Returns:
            PIL Image, sharpness adjusted image.
        """

        return util.random_sharpness(img, self.degrees)


class RandomSolarize(TensorOperation):
    """
    Randomly selects a subrange within the specified threshold range and sets the pixel value within
    the subrange to (255 - pixel).

    Args:
        threshold (tuple, optional): Range of random solarize threshold (default=(0, 255)).
            Threshold values should always be in (min, max) format,
            where min and max are integers in the range [0, 255], and min <= max.
            If min=max, then invert all pixel values above min(max).

    Raises:
        TypeError : If `threshold` is not of type tuple.
        ValueError: If `threshold` is not in range of [0, 255].

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [vision.Decode(), vision.RandomSolarize(threshold=(10,100))]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_random_solarize
    def __init__(self, threshold=(0, 255)):
        super().__init__()
        self.threshold = threshold
        self.implementation = Implementation.C

    def parse(self):
        return cde.RandomSolarizeOperation(self.threshold)


class RandomVerticalFlip(TensorOperation, PyTensorOperation):
    """
    Randomly flip the input image vertically with a given probability.

    Args:
        prob (float, optional): Probability of the image being flipped (default=0.5).

    Raises:
        TypeError: If `prob` is not of type float.
        ValueError: If `prob` is not in range [0, 1].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [vision.Decode(), vision.RandomVerticalFlip(0.25)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_prob
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def parse(self):
        return cde.RandomVerticalFlipOperation(self.prob)

    def execute_pil(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be vertically flipped.

        Returns:
            PIL Image, randomly vertically flipped image.
        """
        return util.random_vertical_flip(img, self.prob)


class RandomVerticalFlipWithBBox(TensorOperation):
    """
    Flip the input image vertically, randomly with a given probability and adjust bounding boxes accordingly.

    Args:
        prob (float, optional): Probability of the image being flipped (default=0.5).

    Raises:
        TypeError: If `prob` is not of type float.
        ValueError: If `prob` is not in range [0, 1].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [vision.Decode(), vision.RandomVerticalFlipWithBBox(0.20)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_prob
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob
        self.implementation = Implementation.C

    def parse(self):
        return cde.RandomVerticalFlipWithBBoxOperation(self.prob)


class Rescale(TensorOperation):
    """
    Rescale the input image with the given rescale and shift. This operator will rescale the input image
    with: output = image * rescale + shift.

    Args:
        rescale (float): Rescale factor.
        shift (float): Shift factor.

    Raises:
        TypeError: If `rescale` is not of type float.
        TypeError: If `shift` is not of type float.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [vision.Decode(), vision.Rescale(1.0 / 255.0, -1.0)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_rescale
    def __init__(self, rescale, shift):
        super().__init__()
        self.rescale = rescale
        self.shift = shift
        self.implementation = Implementation.C

    def parse(self):
        return cde.RescaleOperation(self.rescale, self.shift)


class Resize(TensorOperation, PyTensorOperation):
    """
    Resize the input image to the given size with a given interpolation mode.

    Args:
        size (Union[int, Sequence[int]]): The output size of the resized image. The size value(s) must be positive.
            If size is an integer, the smaller edge of the image will be resized to this value with
            the same image aspect ratio.
            If size is a sequence of length 2, it should be (height, width).
        interpolation (Inter, optional): Image interpolation mode (default=Inter.BILINEAR).
           It can be any of [Inter.BILINEAR, Inter.LINEAR, Inter.NEAREST, Inter.BICUBIC, Inter.AREA, Inter.PILCUBIC,
           Inter.ANTIALIAS].

            - Inter.BILINEAR, means interpolation method is bilinear interpolation.

            - Inter.LINEAR, means interpolation method is bilinear interpolation, here is the same as Inter.BILINEAR.

            - Inter.NEAREST, means interpolation method is nearest-neighbor interpolation.

            - Inter.BICUBIC, means interpolation method is bicubic interpolation.

            - Inter.AREA, means interpolation method is pixel area interpolation.

            - Inter.PILCUBIC, means interpolation method is bicubic interpolation like implemented in Pillow, input
              should be in 3 channels format.

            - Inter.ANTIALIAS, means antialias interpolation.

    Raises:
        TypeError: If `size` is not of type int or Sequence[int].
        TypeError: If `interpolation` is not of type Inter.
        ValueError: If `size` is not positive.
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.vision import Inter
        >>> decode_op = vision.Decode()
        >>> resize_op = vision.Resize([100, 75], Inter.BICUBIC)
        >>> transforms_list = [decode_op, resize_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_resize_interpolation
    def __init__(self, size, interpolation=Inter.LINEAR):
        super().__init__()
        self.py_size = size
        if isinstance(size, int):
            size = (size,)
        self.c_size = size
        self.c_interpolation = None
        self.py_interpolation = None
        if interpolation in [Inter.AREA, Inter.PILCUBIC]:
            self.c_interpolation = Inter.to_c_type(interpolation)
        elif interpolation == Inter.ANTIALIAS:
            self.py_interpolation = Inter.to_python_type(interpolation)
        else:
            self.c_interpolation = Inter.to_c_type(interpolation)
            self.py_interpolation = Inter.to_python_type(interpolation)
        self.random = False

    def parse(self):
        return cde.ResizeOperation(self.c_size, self.c_interpolation)

    def execute_pil(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be resized.

        Returns:
            PIL Image, resized image.
        """
        return util.resize(img, self.py_size, self.py_interpolation)


class ResizeWithBBox(TensorOperation):
    """
    Resize the input image to the given size and adjust bounding boxes accordingly.

    Args:
        size (Union[int, Sequence[int]]): The output size of the resized image.
            If size is an integer, smaller edge of the image will be resized to this value with
            the same image aspect ratio.
            If size is a sequence of length 2, it should be (height, width).
        interpolation (Inter mode, optional): Image interpolation mode (default=Inter.LINEAR).
            It can be any of [Inter.LINEAR, Inter.NEAREST, Inter.BICUBIC].

            - Inter.LINEAR, means interpolation method is bilinear interpolation.

            - Inter.NEAREST, means interpolation method is nearest-neighbor interpolation.

            - Inter.BICUBIC, means interpolation method is bicubic interpolation.

    Raises:
        TypeError: If `size` is not of type int or Sequence[int].
        TypeError: If `interpolation` is not of type Inter.
        ValueError: If `size` is not positive.
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.vision import Inter
        >>> decode_op = vision.Decode()
        >>> bbox_op = vision.ResizeWithBBox(50, Inter.NEAREST)
        >>> transforms_list = [decode_op, bbox_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_resize_interpolation
    def __init__(self, size, interpolation=Inter.LINEAR):
        super().__init__()
        self.size = size
        self.interpolation = interpolation
        self.implementation = Implementation.C

    def parse(self):
        size = self.size
        if isinstance(size, int):
            size = (size,)
        return cde.ResizeWithBBoxOperation(size, Inter.to_c_type(self.interpolation))


class RgbToHsv(PyTensorOperation):
    """
    Convert the input numpy.ndarray images from RGB to HSV.

    Args:
        is_hwc (bool): If True, means the input image is in shape of (H, W, C) or (N, H, W, C).
            Otherwise, it is in shape of (C, H, W) or (N, C, H, W). Default: False.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.transforms import Compose
        >>>
        >>> transforms_list = Compose([vision.Decode(to_pil=True)),
        ...                            vision.CenterCrop(20),
        ...                            vision.ToTensor(),
        ...                            vision.RgbToHsv()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_rgb_to_hsv
    def __init__(self, is_hwc=False):
        super().__init__()
        self.is_hwc = is_hwc
        self.random = False
        self.implementation = Implementation.PY

    def __call__(self, rgb_imgs):
        """
        Call method.

        Args:
            rgb_imgs (numpy.ndarray): RGB images to be converted.

        Returns:
            numpy.ndarray, converted HSV images.
        """
        return util.rgb_to_hsvs(rgb_imgs, self.is_hwc)


class Rotate(TensorOperation):
    """
    Rotate the input image by specified degrees.

    Args:
        degrees (Union[int, float]): Rotation degrees.

        resample (Inter mode, optional): An optional resampling filter (default=Inter.NEAREST).
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
        fill_value (Union[int, tuple[int]], optional): Optional fill color for the area outside the rotated image.
            If it is a 3-tuple, it is used to fill R, G, B channels respectively.
            If it is an integer, it is used for all RGB channels.
            The fill_value values must be in range [0, 255] (default=0).

    Raises:
        TypeError: If `degrees` is not of type integer, float or sequence.
        TypeError: If `resample` is not of type Inter.
        TypeError: If `expand` is not of type bool.
        TypeError: If `center` is not of type tuple.
        TypeError: If `fill_value` is not of type int or tuple[int].
        ValueError: If `fill_value` is not in range [0, 255].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.vision import Inter
        >>> transforms_list = [vision.Decode(),
        ...                    vision.Rotate(degrees=30.0,
        ...                    resample=Inter.NEAREST,
        ...                    expand=True)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @check_rotate
    def __init__(self, degrees, resample=Inter.NEAREST, expand=False, center=None, fill_value=0):
        super().__init__()
        if isinstance(degrees, (int, float)):
            degrees = degrees % 360
        if center is None:
            center = ()
        if isinstance(fill_value, int):
            fill_value = tuple([fill_value] * 3)
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill_value = fill_value
        self.implementation = Implementation.C

    def parse(self):
        return cde.RotateOperation(self.degrees, Inter.to_c_type(self.resample), self.expand, self.center,
                                   self.fill_value)


class SlicePatches(TensorOperation):
    """
    Slice Tensor to multiple patches in horizontal and vertical directions.

    The usage scenario is suitable to large height and width Tensor. The Tensor
    will keep the same if set both num_height and num_width to 1. And the
    number of output tensors is equal to num_height*num_width.

    Args:
        num_height (int, optional): The number of patches in vertical direction, which must be positive (default=1).
        num_width (int, optional): The number of patches in horizontal direction, which must be positive (default=1).
        slice_mode (Inter, optional): A mode represents pad or drop (default=SliceMode.PAD).
            It can be any of [SliceMode.PAD, SliceMode.DROP].
        fill_value (int, optional): The border width in number of pixels in
            right and bottom direction if slice_mode is set to be SliceMode.PAD.
            The fill_value must be in range [0, 255] (default=0).

    Raises:
        TypeError: If `num_height` is not of type integer.
        TypeError: If `num_width` is not of type integer.
        TypeError: If `slice_mode` is not of type Inter.
        TypeError: If `fill_value` is not of type integer.
        ValueError: If `num_height` is not positive.
        ValueError: If `num_width` is not positive.
        ValueError: If `fill_value` is not in range [0, 255].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> # default padding mode
        >>> decode_op = vision.Decode()
        >>> num_h, num_w = (1, 4)
        >>> slice_patches_op = vision.SlicePatches(num_h, num_w)
        >>> transforms_list = [decode_op, slice_patches_op]
        >>> cols = ['img' + str(x) for x in range(num_h*num_w)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"],
        ...                                                 output_columns=cols, column_order=cols)
    """

    @check_slice_patches
    def __init__(self, num_height=1, num_width=1, slice_mode=SliceMode.PAD, fill_value=0):
        super().__init__()
        self.num_height = num_height
        self.num_width = num_width
        self.slice_mode = slice_mode
        self.fill_value = fill_value
        self.implementation = Implementation.C

    def parse(self):
        return cde.SlicePatchesOperation(self.num_height, self.num_width,
                                         SliceMode.to_c_type(self.slice_mode), self.fill_value)


class SoftDvppDecodeRandomCropResizeJpeg(TensorOperation):
    """
    A combination of `Crop`, `Decode` and `Resize` using the simulation algorithm of Ascend series chip DVPP module.

    The usage scenario is consistent with SoftDvppDecodeResizeJpeg.
    The input image size should be in range [32*32, 8192*8192].
    The zoom-out and zoom-in multiples of the image length and width should in the range [1/32, 16].
    Only images with an even resolution can be output. The output of odd resolution is not supported.

    Args:
        size (Union[int, Sequence[int]]): The size of the output image. The size value(s) must be positive.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).
        scale (list, tuple, optional): Range [min, max) of respective size of the
            original size to be cropped, which must be non-negative (default=(0.08, 1.0)).
        ratio (list, tuple, optional): Range [min, max) of aspect ratio to be
            cropped, which must be non-negative (default=(3. / 4., 4. / 3.)).
        max_attempts (int, optional): The maximum number of attempts to propose a valid crop_area (default=10).
            If exceeded, fall back to use center_crop instead. The max_attempts value must be positive.

    Raises:
        TypeError: If `size` is not of type integer or Sequence[int].
        TypeError: If `scale` is not of type tuple.
        TypeError: If `ratio` is not of type tuple.
        TypeError: If `max_attempts` is not of type integer.
        ValueError: If `size` is not positive.
        ValueError: If `scale` is negative.
        ValueError: If `ratio` is negative.
        ValueError: If `max_attempts` is not positive.
        RuntimeError: If given tensor is not a 1D sequence.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> # decode, randomly crop and resize image, keeping aspect ratio
        >>> transforms_list1 = [vision.SoftDvppDecodeRandomCropResizeJpeg(90)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list1,
        ...                                                 input_columns=["image"])
        >>> # decode, randomly crop and resize to landscape style
        >>> transforms_list2 = [vision.SoftDvppDecodeRandomCropResizeJpeg((80, 100))]
        >>> image_folder_dataset_1 = image_folder_dataset_1.map(operations=transforms_list2,
        ...                                                     input_columns=["image"])
    """

    @check_soft_dvpp_decode_random_crop_resize_jpeg
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), max_attempts=10):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.max_attempts = max_attempts
        self.implementation = Implementation.C

    def parse(self):
        return cde.SoftDvppDecodeRandomCropResizeJpegOperation(self.size, self.scale, self.ratio, self.max_attempts)


class SoftDvppDecodeResizeJpeg(TensorOperation):
    """
    Decode and resize JPEG image using the simulation algorithm of Ascend series chip DVPP module.

    It is recommended to use this algorithm in the following scenarios:
    When training, the DVPP of the Ascend chip is not used,
    and the DVPP of the Ascend chip is used during inference,
    and the accuracy of inference is lower than the accuracy of training;
    and the input image size should be in range [32*32, 8192*8192].
    The zoom-out and zoom-in multiples of the image length and width should in the range [1/32, 16].
    Only images with an even resolution can be output. The output of odd resolution is not supported.

    Args:
        size (Union[int, Sequence[int]]): The output size of the resized image. The size value(s) must be positive.
            If size is an integer, smaller edge of the image will be resized to this value with
            the same image aspect ratio.
            If size is a sequence of length 2, it should be (height, width).

    Raises:
        TypeError: If `size` is not of type integer or sequence of integer.
        ValueError: If `size` is not positive.
        RuntimeError: If given tensor is not a 1D sequence.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> # decode and resize image, keeping aspect ratio
        >>> transforms_list1 = [vision.SoftDvppDecodeResizeJpeg(70)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list1,
        ...                                                 input_columns=["image"])
        >>> # decode and resize to portrait style
        >>> transforms_list2 = [vision.SoftDvppDecodeResizeJpeg((80, 60))]
        >>> image_folder_dataset_1 = image_folder_dataset_1.map(operations=transforms_list2,
        ...                                                     input_columns=["image"])
    """

    @check_resize
    def __init__(self, size):
        super().__init__()
        if isinstance(size, int):
            size = (size,)
        self.size = size
        self.implementation = Implementation.C

    def parse(self):
        return cde.SoftDvppDecodeResizeJpegOperation(self.size)


class TenCrop(PyTensorOperation):
    """
    Crop the given image into one central crop and four corners with the flipped version of these.

    Args:
        size (Union[int, sequence]): The size of the cropped image.
            If a single integer is provided, a square of size (size, size) will be cropped with this value.
            If a sequence of length 2 is provided, an image of size (height, width) will be cropped.
        use_vertical_flip (bool, optional): If True, flip the images vertically. Otherwise, flip them
            horizontally. Default: False.

    Raises:
        TypeError: If `size` is not of type integer or sequence of integer.
        TypeError: If `use_vertical_flip` is not of type boolean.
        ValueError: If `size` is not positive.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy
        >>> from mindspore.dataset.transforms.transforms import Compose
        >>>
        >>> transforms_list = Compose([vision.Decode(to_pil=True)),
        ...                            vision.TenCrop(size=200),
        ...                            # 4D stack of 10 images
        ...                            lambda *images: numpy.stack([vision.ToTensor()(image) for image in images])])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_ten_crop
    def __init__(self, size, use_vertical_flip=False):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.use_vertical_flip = use_vertical_flip
        self.random = False
        self.implementation = Implementation.PY

    def execute_pil(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            tuple, a tuple of 10 PIL Image, in order of top_left, top_right, bottom_left, bottom_right, center
                of the original image and top_left, top_right, bottom_left, bottom_right, center of the flipped image.
        """
        return util.ten_crop(img, self.size, self.use_vertical_flip)


class ToPIL(PyTensorOperation):
    """
    Convert the input decoded numpy.ndarray image to PIL Image.

    Note:
        The conversion mode will be determined by the data type using :class:`PIL.Image.fromarray`.

    Raises:
        TypeError: If the input image is not of type :class:`numpy.ndarray` or :class:`PIL.Image`.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.transforms import Compose
        >>>
        >>> # data is already decoded, but not in PIL Image format
        >>> transforms_list = Compose([vision.ToPIL(),
        ...                            vision.RandomHorizontalFlip(0.5),
        ...                            vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    def __init__(self):
        super().__init__()
        self.random = False
        self.implementation = Implementation.PY

    def __call__(self, img):
        """
        Call method.

        Args:
            img (numpy.ndarray): Decoded numpy.ndarray image to be converted to PIL Image.

        Returns:
            PIL Image, converted PIL Image.
        """
        return util.to_pil(img)


class ToTensor(TensorOperation, PyTensorOperation):
    """
    Rescale of pixel value range from [0, 255] to [0.0, 1.0] and change the shape from (H, W, C) to (C, H, W).
    For Python implementation of operation, convert the input PIL Image or numpy.ndarray to numpy.ndarray
    of the desired dtype.

    Args:
        output_type (numpy.dtype, optional): The desired dtype of the output image. Default: :class:`numpy.float32`.

    Raises:
        TypeError: If the input image is not of type :class:`PIL.Image` or :class:`numpy.ndarray`.
        TypeError: If dimension of the input image is not 2 or 3.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.transforms import Compose
        >>>
        >>> # create a list of transformations to be applied to the "image" column of each data row
        >>> transforms_list = Compose([vision.Decode(to_pil=True)),
        ...                            vision.RandomHorizontalFlip(0.5),
        ...                            vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    def __init__(self, output_type="float32"):
        super().__init__()
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

    def parse(self):
        return cde.ToTensorOperation(self.output_type)


class ToType(TypeCast):
    """
    Convert the input numpy.ndarray image to the desired dtype.

    Args:
        output_type (numpy.dtype): The desired dtype of the output image, e.g. :class:`numpy.float32`.

    Raises:
        TypeError: If the input image is not of type :class:`numpy.ndarray`.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore.dataset.transforms.transforms import Compose
        >>>
        >>> transforms_list = Compose([vision.Decode(to_pil=True)),
        ...                            vision.RandomHorizontalFlip(0.5),
        ...                            vision.ToTensor(),
        ...                            vision.ToType(np.float32)])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """


class UniformAugment(CompoundOperation):
    """
    Uniformly select a number of transformations from a sequence and apply them
    sequentially and randomly, which means that there is a chance that a chosen
    transformation will not be applied.

    All transformations in the sequence require the output type to be the same as
    the input. Thus, the latter one can deal with the output of the previous one.

    Args:
         transforms (sequence): Sequence of transformations to select from.
         num_ops (int, optional): Number of transformations to be sequentially and randomly applied. Default: 2.

    Raises:
        TypeError: If `transforms` is not a sequence of data processing operations.
        TypeError: If `num_ops` is not of type integer.
        ValueError: If `num_ops` is not positive.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.transforms import Compose
        >>>
        >>> transforms = [vision.CenterCrop(64),
        ...               vision.RandomColor(),
        ...               vision.RandomSharpness(),
        ...               vision.RandomRotation(30)]
        >>> transforms_list = Compose([vision.Decode(to_pil=True)),
        ...                            vision.UniformAugment(transforms),
        ...                            vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @check_uniform_augment_cpp
    def __init__(self, transforms, num_ops=2):
        super().__init__(transforms)
        self.num_ops = num_ops
        self.random = True

    def parse(self):
        operations = self.parse_transforms()
        return cde.UniformAugOperation(operations, self.num_ops)

    def execute_pil(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image, transformed image.
        """
        return util.uniform_augment(img, self.transforms.copy(), self.num_ops)


class VerticalFlip(TensorOperation):
    """
    Flip the input image vertically.

    Raises:
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [vision.Decode(), vision.VerticalFlip()]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    def __init__(self):
        super().__init__()
        self.implementation = Implementation.C

    def parse(self):
        return cde.VerticalFlipOperation()


def not_random(func):
    """
    Specify the function as "not random", i.e., it produces deterministic result.
    A Python function can only be cached after it is specified as "not random".
    """
    func.random = False
    return func
