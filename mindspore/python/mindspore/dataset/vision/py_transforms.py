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
The module vision.py_transforms is mainly implemented based on Python PIL, which
provides many kinds of image augmentation methods and conversion methods between
PIL.Image.Image and numpy.ndarray. For users who prefer using Python PIL in computer vision
tasks, this module is a good choice to process images. Users can also self-define
their own augmentation methods with Python PIL.
"""
import numbers
import random

import numpy as np

import mindspore.dataset.transforms.py_transforms as py_transforms
from . import py_transforms_util as util
from .c_transforms import parse_padding
from .py_transforms_util import is_pil
from .utils import Border, Inter, ANTIALIAS, CUBIC, LINEAR, NEAREST
from .validators import check_adjust_gamma, check_alpha, check_auto_contrast, check_center_crop, check_cutout, \
    check_five_crop, check_hsv_to_rgb, check_linear_transform, check_mix_up, check_normalize_py, \
    check_normalizepad_py, check_num_channels, check_pad, check_positive_degrees, check_prob, check_random_affine, \
    check_random_color_adjust, check_random_crop, check_random_erasing, check_random_perspective, \
    check_random_resize_crop, check_random_rotation, check_resize_interpolation, check_rgb_to_bgr, check_rgb_to_hsv, \
    check_ten_crop, check_uniform_augment_py, deprecated_py_vision

DE_PY_BORDER_TYPE = {Border.CONSTANT: 'constant',
                     Border.EDGE: 'edge',
                     Border.REFLECT: 'reflect',
                     Border.SYMMETRIC: 'symmetric'}

DE_PY_INTER_MODE = {Inter.NEAREST: NEAREST,
                    Inter.ANTIALIAS: ANTIALIAS,
                    Inter.LINEAR: LINEAR,
                    Inter.CUBIC: CUBIC}


class AdjustGamma(py_transforms.PyTensorOperation):
    """
    Perform gamma correction on the input PIL Image.

    Args:
        gamma (float): The gamma parameter in correction equation, must be non negative.
        gain (float, optional): The constant multiplier. Default: ``1.0``.

    Raises:
        TypeError: If `gain` is not of type float.
        TypeError: If `gamma` is not of type float.
        ValueError: If `gamma` is less than 0.
        RuntimeError: If shape of the input image is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.AdjustGamma(gamma=10.0),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
    @check_adjust_gamma
    def __init__(self, gamma, gain=1.0):
        self.gamma = gamma
        self.gain = gain
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL.Image.Image): Image to be gamma adjusted.

        Returns:
            PIL.Image.Image, gamma adjusted image.
        """

        return util.adjust_gamma(img, self.gamma, self.gain)


class AutoContrast(py_transforms.PyTensorOperation):
    """
    Maximize (normalize) contrast of the input PIL Image.

    It will first calculate a histogram of the input image, remove `cutoff` percent of the
    lightest and darkest pixels from the histogram, then remap the pixel value to [0, 255],
    making the darkest pixel black and the lightest pixel white.

    Args:
        cutoff (float, optional): Percent to cut off from the histogram on the low and
            high ends, must be in range of [0.0, 50.0]. Default: ``0.0``.
        ignore (Union[int, Sequence[int]], optional): Background pixel value, which will be
            directly remapped to white. Default: ``None``, means no background.

    Raises:
        TypeError: If `cutoff` is not of type float.
        TypeError: If `ignore` is not of type int or sequence.
        ValueError: If `cutoff` is not in range [0, 50.0).
        ValueError: If `ignore` is not in range [0, 255].
        RuntimeError: If shape of the input image is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.AutoContrast(),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
    @check_auto_contrast
    def __init__(self, cutoff=0.0, ignore=None):
        self.cutoff = cutoff
        self.ignore = ignore
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL.Image.Image): Image to be automatically contrasted.

        Returns:
            PIL.Image.Image, automatically contrasted image.
        """

        return util.auto_contrast(img, self.cutoff, self.ignore)


class CenterCrop(py_transforms.PyTensorOperation):
    """
    Crop the central region of the input PIL Image with the given size.

    Args:
        size (Union[int, Sequence[int, int]]): The size of the cropped image.
            If int is provided, a square of size `(size, size)` will be cropped with this value.
            If Sequence[int, int] is provided, its two elements will be taken as the cropped height and width.

    Raises:
        TypeError: If `size` is not of type int or Sequence[int, int].
        ValueError: If `size` is not positive.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.CenterCrop(64),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
    @check_center_crop
    def __init__(self, size):
        self.size = size
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL.Image.Image): Image to be center cropped.

        Returns:
            PIL.Image.Image, cropped image.
        """
        return util.center_crop(img, self.size)


class Cutout(py_transforms.PyTensorOperation):
    """
    Randomly cut out a certain number of square patches on the input numpy.ndarray image,
    setting the pixel values in the patch to zero.

    See `Improved Regularization of Convolutional Neural Networks with Cutout <https://arxiv.org/pdf/1708.04552.pdf>`_ .

    Args:
        length (int): The side length of square patches to be cut out.
        num_patches (int, optional): The number of patches to be cut out. Default: ``1``.

    Raises:
        TypeError: If `length` is not of type int.
        TypeError: If `num_patches` is not of type int.
        ValueError: If `length` is less than or equal 0.
        ValueError: If `num_patches` is less than or equal 0.
        RuntimeError: If shape of the input image is not <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.ToTensor(),
        ...                            py_vision.Cutout(80)])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision("CutOut")
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
        if self.length > image_h or self.length > image_w:
            raise ValueError(
                f"Patch length is too large, got patch length: {self.length} and image height: {image_h}, image "
                f"width: {image_w}")

        scale = (self.length * self.length) / (image_h * image_w)
        bounded = False

        for _ in range(self.num_patches):
            i, j, erase_h, erase_w, erase_value = util.get_erase_params(np_img, (scale, scale), (1, 1), 0, bounded,
                                                                        1)
            np_img = util.erase(np_img, i, j, erase_h, erase_w, erase_value)
        return np_img


class Decode(py_transforms.PyTensorOperation):
    """
    Decode the input raw image bytes to PIL Image format in RGB mode.

    Raises:
        ValueError: If the input is not raw image bytes.
        ValueError: If the input image is already decoded.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomHorizontalFlip(0.5),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
    def __init__(self):
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (Bytes-like Object): Raw image data to be decoded.

        Returns:
            PIL.Image.Image, decoded PIL Image in RGB mode.
        """
        return util.decode(img)


class Equalize(py_transforms.PyTensorOperation):
    """
    Equalize the histogram of the input PIL Image.

    By applying a non-linear mapping to the input image, it creates a uniform
    distribution of grayscale values in the output.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.Equalize(),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
    def __init__(self):
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL.Image.Image): Image to be equalized.

        Returns:
            PIL.Image.Image, equalized image.
        """

        return util.equalize(img)


class FiveCrop(py_transforms.PyTensorOperation):
    """
    Crop the given image into one central crop and four corners.

    Args:
        size (Union[int, Sequence[int, int]]): The size of the cropped image.
            If int is provided, a square of size `(size, size)` will be cropped with this value.
            If Sequence[int, int] is provided, its two elements will be taken as the cropped height and width.

    Raises:
        TypeError: If `size` is not of type int or Sequence[int, int].
        ValueError: If `size` is not positive.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.FiveCrop(size=200),
        ...                            # 4D stack of 5 images
        ...                            lambda *images: numpy.stack([py_vision.ToTensor()(image) for image in images])])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
    @check_five_crop
    def __init__(self, size):
        self.size = size
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL.Image.Image): Image to be cropped.

        Returns:
            tuple[PIL.Image.Image], five cropped images in order of top_left, top_right, bottom_left,
            bottom_right and center.
        """
        return util.five_crop(img, self.size)


class Grayscale(py_transforms.PyTensorOperation):
    """
    Convert the input PIL Image to grayscale.

    Args:
        num_output_channels (int): The number of channels desired for the output image, must be ``1`` or ``3``.
            If ``3`` is provided, the returned image will have 3 identical RGB channels. Default: ``1``.

    Raises:
        TypeError: If `num_output_channels` is not of type int.
        ValueError: If `num_output_channels` is not ``1`` or ``3``.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.Grayscale(3),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
    @check_num_channels
    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL.Image.Image): Image to be converted to grayscale.

        Returns:
            PIL.Image.Image, converted grayscale image.
        """
        return util.grayscale(img, num_output_channels=self.num_output_channels)


class HsvToRgb(py_transforms.PyTensorOperation):
    """
    Convert the input numpy.ndarray images from HSV to RGB.

    Args:
        is_hwc (bool): If ``True``, means the input image is in shape of (H, W, C) or (N, H, W, C).
            Otherwise, it is in shape of (C, H, W) or (N, C, H, W). Default: ``False``.

    Raises:
        TypeError: If `is_hwc` is not of type bool.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.CenterCrop(20),
        ...                            py_vision.ToTensor(),
        ...                            py_vision.HsvToRgb()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
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


class HWC2CHW(py_transforms.PyTensorOperation):
    """
    Transpose the input numpy.ndarray image from shape (H, W, C) to (C, H, W).
    If the input image is of shape <H, W>, it will remain unchanged.

    Raises:
        TypeError: If the input image is not of type :class:`numpy.ndarray` .
        TypeError: If shape of the input image is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([c_vision.Decode(),
        ...                            py_vision.HWC2CHW()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
    def __init__(self):
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (numpy.ndarray): numpy.ndarray to be transposed.

        Returns:
            numpy.ndarray, transposed numpy.ndarray.
        """
        return util.hwc_to_chw(img)


class Invert(py_transforms.PyTensorOperation):
    """
    Invert the colors of the input PIL Image.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.Invert(),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
    def __init__(self):
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL.Image.Image): Image to be color inverted.

        Returns:
            PIL.Image.Image, color inverted image.
        """

        return util.invert_color(img)


class LinearTransformation(py_transforms.PyTensorOperation):
    r"""
    Linearly transform the input numpy.ndarray image with a square transformation matrix and a mean vector.

    It will first flatten the input image and subtract the mean vector from it, then compute the dot
    product with the transformation matrix, finally reshape it back to its original shape.

    Args:
        transformation_matrix (numpy.ndarray): A square transformation matrix in shape of (D, D), where
            :math:`D = C \times H \times W` .
        mean_vector (numpy.ndarray): A mean vector in shape of (D,), where :math:`D = C \times H \times W` .

    Raises:
        TypeError: If `transformation_matrix` is not of type :class:`numpy.ndarray` .
        TypeError: If `mean_vector` is not of type :class:`numpy.ndarray` .

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
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

    @deprecated_py_vision()
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


class MixUp(py_transforms.PyTensorOperation):
    """
    Randomly mix up a batch of images together with its labels.

    Each image will be multiplied by a random weight :math:`lambda` generated from the Beta distribution and then added
    to another image multiplied by :math:`1 - lambda` . The same transformation will be applied to their labels with the
    same value of :math:`lambda` . Make sure that the labels are one-hot encoded in advance.

    Args:
        batch_size (int): The number of images in a batch.
        alpha (float): The alpha and beta parameter for the Beta distribution.
        is_single (bool, optional): If ``True``, it will randomly mix up [img0, ..., img(n-1), img(n)] with
            [img1, ..., img(n), img0] in each batch. Otherwise, it will randomly mix up images with the
            output of the previous batch. Default: ``True``.

    Raises:
        TypeError: If `batch_size` is not of type int.
        TypeError: If `alpha` is not of type float.
        TypeError: If `is_single` is not of type bool.
        ValueError: If `batch_size` is not positive.
        ValueError: If `alpha` is not positive.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> # first decode the image
        >>> image_folder_dataset = image_folder_dataset.map(operations=c_vision.Decode(),
        ...                                                 input_columns="image")
        >>> # then ont hot decode the label
        >>> image_folder_dataset = image_folder_dataset.map(operations=c_transforms.OneHot(10),
        ...                                                 input_columns="label")
        >>> # batch the samples
        >>> batch_size = 4
        >>> image_folder_dataset = image_folder_dataset.batch(batch_size=batch_size)
        >>> # finally mix up the images and labels
        >>> image_folder_dataset = image_folder_dataset.map(
        ...     operations=py_vision.MixUp(batch_size=batch_size, alpha=0.2),
        ...     input_columns=["image", "label"])
    """

    @deprecated_py_vision()
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


class Normalize(py_transforms.PyTensorOperation):
    r"""
    Normalize the input numpy.ndarray image of shape (C, H, W) with the specified mean and standard deviation.

    .. math::

        output_{c} = \frac{input_{c} - mean_{c}}{std_{c}}

    Note:
        The pixel values of the input image need to be in range of [0.0, 1.0].
        If not so, please call :class:`mindspore.dataset.vision.py_transforms.ToTensor` first.

    Args:
        mean (Union[float, Sequence[float]]): Mean pixel values for each channel,
            must be in range of [0.0, 1.0].
            If float is provided, it will be applied to each channel.
            If Sequence[float] is provided, it should have the same length with channel
            and be arranged in channel order.
        std (Union[float, Sequence[float]]): Standard deviation values for each channel, must be in range of (0.0, 1.0].
            If float is provided, it will be applied to each channel.
            If Sequence[float] is provided, it should have the same length with channel
            and be arranged in channel order.

    Raises:
        TypeError: If the input image is not of type :class:`numpy.ndarray` .
        TypeError: If dimension of the input image is not 3.
        NotImplementedError: If dtype of the input image is int.
        ValueError: If lengths of `mean` and `std` are not equal.
        ValueError: If length of `mean` or `std` is neither equal to 1 nor equal to the length of channel.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomHorizontalFlip(0.5),
        ...                            py_vision.ToTensor(),
        ...                            py_vision.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262))])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
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
        The pixel values of the input image need to be in range of [0.0, 1.0].
        If not so, please call :class:`mindspore.dataset.vision.py_transforms.ToTensor` first.

    Args:
        mean (Union[float, Sequence[float]]): Mean pixel values for each channel, must be in range of [0.0, 1.0].
            If float is provided, it will be applied to each channel.
            If Sequence[float] is provided, it should have the same length with channel
            and be arranged in channel order.
        std (Union[float, Sequence[float]]): Standard deviation values for each channel, must be in range of (0.0, 1.0].
            If float is provided, it will be applied to each channel.
            If Sequence[float] is provided, it should have the same length with channel
            and be arranged in channel order.
        dtype (str): The dtype of the output image. Only ``"float32"`` and ``"float16"`` are supported.
            Default: ``"float32"``.

    Raises:
        TypeError: If the input image is not of type :class:`numpy.ndarray` .
        TypeError: If dimension of the input image is not 3.
        NotImplementedError: If dtype of the input image is int.
        ValueError: If lengths of `mean` and `std` are not equal.
        ValueError: If length of `mean` or `std` is neither equal to 1 nor equal to the length of channel.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomHorizontalFlip(0.5),
        ...                            py_vision.ToTensor(),
        ...                            py_vision.NormalizePad((0.491, 0.482, 0.447), (0.247, 0.243, 0.262), "float32")])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
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


class Pad(py_transforms.PyTensorOperation):
    """
    Pad the input PIL Image on all sides.

    Args:
        padding (Union[int, Sequence[int, int], Sequence[int, int, int, int]]): The number of pixels to pad
            on each border.
            If int is provided, pad all borders with this value.
            If Sequence[int, int] is provided, pad the left and top borders with the
            first value and the right and bottom borders with the second value.
            If Sequence[int, int, int, int] is provided, pad the left, top, right and bottom borders respectively.
        fill_value (Union[int, tuple[int, int, int]], optional): Pixel value used to pad the borders,
            only valid when `padding_mode` is ``Border.CONSTANT``.
            If int is provided, it will be used for all RGB channels.
            If tuple[int, int, int] is provided, it will be used for R, G, B channels respectively. Default: ``0``.
        padding_mode (Border, optional): Method of padding. It can be ``Border.CONSTANT``, ``Border.EDGE``,
            ``Border.REFLECT`` or ``Border.SYMMETRIC``. Default: ``Border.CONSTANT``.

            - ``Border.CONSTANT`` , pads with a constant value.
            - ``Border.EDGE`` , pads with the last value at the edge of the image.
            - ``Border.REFLECT`` , pads with reflection of the image omitting the last value on the edge.
            - ``Border.SYMMETRIC`` , pads with reflection of the image repeating the last value on the edge.

    Note:
        The behavior when `padding` is a sequence of length 2 will change from padding left/top with
        the first value and right/bottom with the second, to padding left/right with the first one
        and top/bottom with the second in the future. Or you can pass in a 4-element sequence to specify
        left, top, right and bottom respectively.

    Raises:
        TypeError: If `padding` is not of type int or Sequence[int, int].
        TypeError: If `fill_value` is not of type int or tuple[int, int, int].
        TypeError: If `padding_mode` is not of type :class:`mindspore.dataset.vision.Border` .
        ValueError: If `padding` is negative.
        ValueError: If `fill_value` is not in range of [0, 255].
        RuntimeError: If shape of the input image is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            # adds 10 pixels (default black) to each border of the image
        ...                            py_vision.Pad(padding=10),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
    @check_pad
    def __init__(self, padding, fill_value=0, padding_mode=Border.CONSTANT):
        self.padding = parse_padding(padding)
        self.fill_value = fill_value
        self.padding_mode = DE_PY_BORDER_TYPE.get(padding_mode)
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL.Image.Image): Image to be padded.

        Returns:
            PIL.Image.Image, padded image.
        """
        return util.pad(img, self.padding, self.fill_value, self.padding_mode)


class RandomAffine(py_transforms.PyTensorOperation):
    """
    Apply random affine transformation to the input PIL Image.

    Args:
        degrees (Union[float, Sequence[float, float]]): Range of degrees to select from.
            If float is provided, the degree will be randomly selected from ( `-degrees` , `degrees` ).
            If Sequence[float, float] is provided, it needs to be arranged in order of (min, max).
        translate (Sequence[float, float], optional): Maximum absolute fraction sequence in shape of (tx, ty)
            for horizontal and vertical translations. The horizontal and vertical shifts are randomly
            selected from (-tx * width, tx * width) and (-ty * height, ty * height) respectively.
            Default: ``None``, means no translation.
        scale (Sequence[float, float], optional): Range of scaling factor to select from.
            Default: ``None``, means to keep the original scale.
        shear (Union[float, Sequence[float, float], Sequence[float, float, float, float]], optional):
            Range of shear factor to select from.
            If float is provided, a shearing parallel to X axis with a factor selected from
            ( `-shear` , `shear` ) will be applied.
            If Sequence[float, float] is provided, a shearing parallel to X axis with a factor selected
            from ( `shear` [0], `shear` [1]) will be applied.
            If Sequence[float, float, float, float] is provided, a shearing parallel to X axis with a factor selected
            from ( `shear` [0], `shear` [1]) and a shearing parallel to Y axis with a factor selected from
            ( `shear` [2], `shear` [3]) will be applied. Default: ``None``, means no shearing.
        resample (Inter, optional): Method of interpolation. It can be ``Inter.BILINEAR``, ``Inter.NEAREST``
            or ``Inter.BICUBIC``. If the input PIL Image is in mode of "1" or "P", ``Inter.NEAREST`` will be
            used directly. Default: ``Inter.NEAREST``.

            - ``Inter.BILINEA`` , bilinear interpolation.
            - ``Inter.NEAREST`` , nearest-neighbor interpolation.
            - ``Inter.BICUBIC`` , bicubic interpolation.

        fill_value (Union[int, tuple[int, int, int]], optional): Pixel value for areas outside the transform image.
            If int is provided, it will be used for all RGB channels.
            If tuple[int, int, int] is provided, it will be used for R, G, B channels respectively.
            Only supported with Pillow 5.0.0 and above. Default: ``0``.

    Raises:
        TypeError: If `degrees` is not of type float or Sequence[float, float].
        TypeError: If `translate` is not of type Sequence[float, float].
        TypeError: If `scale` is not of type Sequence[float, float].
        TypeError: If `shear` is not of type float or Sequence[float, float].
        TypeError: If `resample` is not of type :class:`mindspore.dataset.vision.Inter` .
        TypeError: If `fill_value` is not of type int or tuple[int, int, int].
        ValueError: If `degrees` is negative.
        ValueError: If `translate` is not in range of [-1.0, 1.0].
        ValueError: If `scale` is negative.
        ValueError: If `shear` is not positive.
        RuntimeError: If shape of the input image is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
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
        self.resample = DE_PY_INTER_MODE.get(resample)
        self.fill_value = fill_value

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL.Image.Image): Image to be randomly affine transformed.

        Returns:
            PIL.Image.Image, randomly affine transformed image.
        """

        return util.random_affine(img,
                                  self.degrees,
                                  self.translate,
                                  self.scale_ranges,
                                  self.shear,
                                  self.resample,
                                  self.fill_value)


class RandomColor(py_transforms.PyTensorOperation):
    """
    Adjust the color balance of the input PIL Image by a random degree.

    Args:
        degrees (Sequence[float, float]): Range of color adjustment degree to select from,
            must be a Sequence of length 2, arranged in order of (min, max).
            A degree of 1.0 gives the original image, a degree of ``0.0`` gives a black and white image
            and higher degrees mean more brightness, contrast, etc. Default: ``(0.1, 1.9)``.

    Raises:
        TypeError: If `degrees` is not of type Sequence[float, float].
        ValueError: If `degrees` is negative.
        RuntimeError: If shape of the input image is not <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomColor((0.5, 2.0)),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
    @check_positive_degrees
    def __init__(self, degrees=(0.1, 1.9)):
        self.degrees = degrees

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL.Image.Image): Image to be color adjusted.

        Returns:
            PIL.Image.Image, color adjusted image.
        """

        return util.random_color(img, self.degrees)


class RandomColorAdjust(py_transforms.PyTensorOperation):
    """
    Randomly adjust the brightness, contrast, saturation, and hue of the input PIL Image.

    Args:
        brightness (Union[float, Sequence[float, float]], optional): Range of brightness adjustment factor
            to select from, must be non negative.
            If float is provided, the factor will be uniformly selected from
            [max(0, 1 - `brightness` ), 1 + `brightness` ).
            If Sequence[float, float] is provided, it should be arranged in order of (min, max). Default: ``(1, 1)``.
        contrast (Union[float, Sequence[float, float]], optional): Range of contrast adjustment factor
            to select from, must be non negative.
            If float is provided, the factor will be uniformly selected from [max(0, 1 - `contrast` ), 1 + `contrast` ).
            If Sequence[float, float] is provided, it should be arranged in order of (min, max). Default: ``(1, 1)``.
        saturation (Union[float, Sequence[float, float]], optional): Range of saturation adjustment factor
            to select from, must be non negative.
            If float is provided, the factor will be uniformly selected from
            [max(0, 1 - `saturation` ), 1 + `saturation` ).
            If Sequence[float, float] is provided, it should be arranged in order of (min, max). Default: ``(1, 1)``.
        hue (Union[float, Sequence[float, float]], optional): Range of hue adjustment factor to select from.
            If float is provided, it must be in range of [0, 0.5], and the factor will be uniformly
            selected from [ `-hue` , `hue` ).
            If Sequence[float, float] is provided, the elements must be in range of [-0.5, 0.5] and arranged in
            order of (min, max). Default: ``(0, 0)``.

    Raises:
        TypeError: If `brightness` is not of type float or Sequence[float, float].
        TypeError: If `contrast` is not of type float or Sequence[float, float].
        TypeError: If `saturation` is not of type float or Sequence[float, float].
        TypeError: If `hue` is not of type float or Sequence[float, float].
        ValueError: If `brightness` is negative.
        ValueError: If `contrast` is negative.
        ValueError: If `saturation` is negative.
        ValueError: If `hue` is not in range of [-0.5, 0.5].

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomColorAdjust(0.4, 0.4, 0.4, 0.1),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
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
            img (PIL.Image.Image): Image to be randomly color adjusted.

        Returns:
            PIL.Image.Image, randomly color adjusted image.
        """
        return util.random_color_adjust(img, self.brightness, self.contrast, self.saturation, self.hue)


class RandomCrop(py_transforms.PyTensorOperation):
    """
    Crop the input PIL Image at a random location with the specified size.

    Args:
        size (Union[int, Sequence[int, int]]): The size of the cropped image.
            If int is provided, a square of size `(size, size)` will be cropped with this value.
            If Sequence[int, int] is provided, its two elements will be taken as the cropped height and width.
        padding (Union[int, Sequence[int, int], Sequence[int, int, int, int]], optional): The number of pixels to pad
            on each border. When specified, it will pad the image before random cropping.
            If int is provided, pad all borders with this value.
            If Sequence[int, int] is provided, pad the left and top borders with the
            first value and the right and bottom borders with the second value.
            If Sequence[int, int, int, int] is provided, pad the left, top, right and bottom borders respectively.
            Default: ``None``, means not to pad.
        pad_if_needed (bool, optional): Whether to pad the image if either side is shorter than
            the given cropping size. Default: ``False``, means not to pad.
        fill_value (Union[int, tuple[int, int, int]], optional): Pixel value used to pad the borders,
            only valid when `padding_mode` is ``Border.CONSTANT``.
            If int is provided, it will be used for all RGB channels.
            If tuple[int, int, int] is provided, it will be used for R, G, B channels respectively. Default: ``0``.
        padding_mode (Border, optional): Method of padding. It can be ``Border.CONSTANT``, ``Border.EDGE``,
            ``Border.REFLECT`` or ``Border.SYMMETRIC``. Default: ``Border.CONSTANT``.

            - ``Border.CONSTANT`` , pads with a constant value.
            - ``Border.EDGE`` , pads with the last value at the edge of the image.
            - ``Border.REFLECT`` , pads with reflection of the image omitting the last value on the edge.
            - ``Border.SYMMETRIC`` , pads with reflection of the image repeating the last value on the edge.

    Note:
        The behavior when `padding` is a sequence of length 2 will change from padding left/top with
        the first value and right/bottom with the second, to padding left/right with the first one
        and top/bottom with the second in the future. Or you can pass in a 4-element sequence to specify
        left, top, right and bottom respectively.

    Raises:
        TypeError: If `size` is not of type int or Sequence[int, int].
        TypeError: If `padding` is not of type int, Sequence[int, int] or Sequence[int, int, int, int].
        TypeError: If `pad_if_needed` is not of type bool.
        TypeError: If `fill_value` is not of type int or tuple[int, int, int].
        TypeError: If `padding_mode` is not of type :class:`mindspore.dataset.vision.Border` .
        ValueError: If `size` is not positive.
        ValueError: If `padding` is negative.
        ValueError: If `fill_value` is not in range of [0, 255].

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomCrop(224),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
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
        self.padding_mode = DE_PY_BORDER_TYPE.get(padding_mode)

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL.Image.Image): Image to be randomly cropped.

        Returns:
            PIL.Image.Image, cropped image.
        """
        return util.random_crop(img, self.size, self.padding, self.pad_if_needed,
                                self.fill_value, self.padding_mode)


class RandomErasing(py_transforms.PyTensorOperation):
    """
    Randomly erase pixels within a random selected rectangle erea on the input numpy.ndarray image.

    See `Random Erasing Data Augmentation <https://arxiv.org/pdf/1708.04896.pdf>`_ .

    Args:
        prob (float, optional): Probability of performing erasing. Default: ``0.5``.
        scale (Sequence[float, float], optional): Range of area scale of the erased area relative
            to the original image to select from, arranged in order of (min, max).
            Default: ``(0.02, 0.33)``.
        ratio (Sequence[float, float], optional): Range of aspect ratio of the erased area to select
            from, arraged in order of (min, max). Default: ``(0.3, 3.3)``.
        value (Union[int, str, Sequence[int, int, int]]): Pixel value used to pad the erased area.
            If int is provided, it will be used for all RGB channels.
            If Sequence[int, int, int] is provided, it will be used for R, G, B channels respectively.
            If a string of ``'random'`` is provided, each pixel will be erased with a random value obtained
            from a standard normal distribution. Default: ``0``.
        inplace (bool, optional): Whether to apply erasing inplace. Default: ``False``.
        max_attempts (int, optional): The maximum number of attempts to propose a valid
            erased area, beyond which the original image will be returned. Default: ``10``.

    Raises:
        TypeError: If `prob` is not of type float.
        TypeError: If `scale` is not of type Sequence[float, float].
        TypeError: If `ratio` is not of type Sequence[float, float].
        TypeError: If `value` is not of type int, str, or Sequence[int, int, int].
        TypeError: If `inplace` is not of type bool.
        TypeError: If `max_attempts` is not of type int.
        ValueError: If `prob` is not in range of [0, 1].
        ValueError: If `scale` is negative.
        ValueError: If `ratio` is negative.
        ValueError: If `value` is not in range of [0, 255].
        ValueError: If `max_attempts` is not positive.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.ToTensor(),
        ...                            py_vision.RandomErasing(value='random')])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
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


class RandomGrayscale(py_transforms.PyTensorOperation):
    """
    Randomly convert the input PIL Image to grayscale.

    Args:
        prob (float, optional): Probability of performing grayscale conversion. Default: ``0.1``.

    Raises:
        TypeError: If `prob` is not of type float.
        ValueError: If `prob` is not in range of [0, 1].

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomGrayscale(0.3),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
    @check_prob
    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL.Image.Image): Image to be randomly converted to grayscale.

        Returns:
            PIL.Image.Image, randomly converted grayscale image, which has the same number of channels
                as the input image.
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


class RandomHorizontalFlip(py_transforms.PyTensorOperation):
    """
    Randomly flip the input PIL Image horizontally with a given probability.

    Args:
        prob (float, optional): Probability of performing horizontally flip. Default: ``0.5``.

    Raises:
        TypeError: If `prob` is not of type float.
        ValueError: If `prob` is not in range of [0, 1].
        RuntimeError: If shape of the input image is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomHorizontalFlip(0.5),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
    @check_prob
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL.Image.Image): Image to be horizontally flipped.

        Returns:
            PIL.Image.Image, randomly horizontally flipped image.
        """
        return util.random_horizontal_flip(img, self.prob)


class RandomLighting(py_transforms.PyTensorOperation):
    """
    Add AlexNet-style PCA-based noise to the input PIL Image.

    Args:
        alpha (float, optional): Intensity of the noise. Default: ``0.05``.

    Raises:
        TypeError: If `alpha` is not of type float.
        ValueError: If `alpha` is negative.
        RuntimeError: If shape of input image is not <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomLighting(0.1),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
    @check_alpha
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL.Image.Image): Image to be added AlexNet-style PCA-based noise.

        Returns:
            PIL.Image.Image, image with noise added.
        """

        return util.random_lighting(img, self.alpha)


class RandomPerspective(py_transforms.PyTensorOperation):
    """
    Randomly apply perspective transformation to the input PIL Image with a given probability.

    Args:
        distortion_scale (float, optional): Scale of distortion, in range of [0, 1]. Default: ``0.5``.
        prob (float, optional): Probability of performing perspective transformation. Default: ``0.5``.
        interpolation (Inter, optional): Method of interpolation. It can be ``Inter.BILINEAR``,
            ``Inter.NEAREST`` or ``Inter.BICUBIC``. Default: ``Inter.BICUBIC``.

            - ``Inter.BILINEA`` , bilinear interpolation.
            - ``Inter.NEAREST`` , nearest-neighbor interpolation.
            - ``Inter.BICUBIC`` , bicubic interpolation.

    Raises:
        TypeError: If `distortion_scale` is not of type float.
        TypeError: If `prob` is not of type float.
        TypeError: If `interpolation` is not of type :class:`mindspore.dataset.vision.Inter` .
        ValueError: If `distortion_scale` is not in range of [0, 1].
        ValueError: If `prob` is not in range of [0, 1].

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomPerspective(prob=0.1),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
    @check_random_perspective
    def __init__(self, distortion_scale=0.5, prob=0.5, interpolation=Inter.BICUBIC):
        self.distortion_scale = distortion_scale
        self.prob = prob
        self.interpolation = DE_PY_INTER_MODE.get(interpolation)

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL.Image.Image): Image to be applied randomly perspective transformation.

        Returns:
            PIL.Image.Image, image applied randomly perspective transformation.
        """
        if not is_pil(img):
            raise ValueError("Input image should be a Pillow image.")
        if self.prob > random.random():
            start_points, end_points = util.get_perspective_params(
                img, self.distortion_scale)
            return util.perspective(img, start_points, end_points, self.interpolation)
        return img


class RandomResizedCrop(py_transforms.PyTensorOperation):
    """
    Randomly crop the input PIL Image and resize it to a given size.

    Args:
        size (Union[int, Sequence[int, int]]): The size of the cropped image.
            If int is provided, a square of size `(size, size)` will be cropped with this value.
            If Sequence[int, int] is provided, its two elements will be taken as the cropped height and width.
        scale (Sequence[float, float], optional): Range of area scale of the cropped area relative
            to the original image to select from, arraged in order or (min, max). Default: ``(0.08, 1.0)``.
        ratio (Sequence[float, float], optional): Range of aspect ratio of the cropped area to select
            from, arraged in order of (min, max). Default: ``(3./4., 4./3.)``.
        interpolation (Inter, optional): Method of interpolation. It can be ``Inter.NEAREST``,
            ``Inter.ANTIALIAS``, ``Inter.BILINEAR`` or ``Inter.BICUBIC``. Default: ``Inter.BILINEAR``.

            - ``Inter.NEAREST`` , nearest-neighbor interpolation.
            - ``Inter.ANTIALIAS`` , antialias interpolation.
            - ``Inter.BILINEA`` , bilinear interpolation.
            - ``Inter.BICUBIC`` , bicubic interpolation.

        max_attempts (int, optional): The maximum number of attempts to propose a valid
            crop area, beyond which it will fall back to use center crop instead. Default: ``10``.

    Raises:
        TypeError: If `size` is not of type int or Sequence[int, int].
        TypeError: If `scale` is not of type Sequence[float, float].
        TypeError: If `ratio` is not of type Sequence[float, float].
        TypeError: If `interpolation` is not of type :class:`mindspore.dataset.vision.Inter` .
        TypeError: If `max_attempts` is not of type int.
        ValueError: If `size` is not positive.
        ValueError: If `scale` is negative.
        ValueError: If `ratio` is negative.
        ValueError: If `max_attempts` is not positive.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomResizedCrop(224),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
    @check_random_resize_crop
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation=Inter.BILINEAR, max_attempts=10):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = DE_PY_INTER_MODE.get(interpolation)
        self.max_attempts = max_attempts

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL.Image.Image): Image to be randomly cropped and resized.

        Returns:
            PIL.Image.Image, randomly cropped and resized image.
        """
        return util.random_resize_crop(img, self.size, self.scale, self.ratio,
                                       self.interpolation, self.max_attempts)


class RandomRotation(py_transforms.PyTensorOperation):
    """
    Rotate the input PIL Image by a random angle.

    Args:
        degrees (Union[float, Sequence[float, float]]): Range of rotation degree to select from.
            If int is provided, the rotation degree will be randomly selected from ( `-degrees` , `degrees` ).
            If Sequence[float, float] is provided, it should be arranged in order of (min, max).
        resample (Inter, optional): Method of interpolation. It can be ``Inter.NEAREST``,
            ``Inter.BILINEAR`` or ``Inter.BICUBIC``. If the input PIL Image is in mode of "1" or "P",
            ``Inter.NEAREST`` will be used directly. Default: ``Inter.NEAREST``.

            - ``Inter.NEAREST`` , nearest-neighbor interpolation.

            - ``Inter.BILINEA`` , bilinear interpolation.

            - ``Inter.BICUBIC`` , bicubic interpolation.

        expand (bool, optional): If ``True``, it will expand the image to make it large enough to hold the entire
            rotated image. If ``False``, keep the image the same size as the input. Please note that the expansion
            assumes rotation around the center and no translation. Default: ``False``.
        center (Sequence[int, int], optional): The position of the rotation center, taking the upper left corner
            as the origin. It should be arranged in order of (width, height). Default: ``None``, means to set the
            center of the image.
        fill_value (Union[int, tuple[int, int, int]], optional): Pixel value for areas outside the rotated image.
            If int is provided, it will be used for all RGB channels.
            If tuple[int, int, int] is provided, it will be used for R, G, B channels respectively. Default: ``0``.

    Raises:
        TypeError: If `degrees` is not of type float or Sequence[float, float].
        TypeError: If `resample` is not of type :class:`mindspore.dataset.vision.Inter` .
        TypeError: If `expand` is not of type bool.
        TypeError: If `center` is not of type Sequence[int, int].
        TypeError: If `fill_value` is not of type int or tuple[int, int, int].
        ValueError: If `fill_value` is not in range of [0, 255].
        RuntimeError: If shape of the input image is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomRotation(30),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
    @check_random_rotation
    def __init__(self, degrees, resample=Inter.NEAREST, expand=False, center=None, fill_value=0):
        self.degrees = degrees
        self.resample = DE_PY_INTER_MODE.get(resample)
        self.expand = expand
        self.center = center
        self.fill_value = fill_value

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL.Image.Image): Image to be randomly rotated.

        Returns:
            PIL.Image.Image, randomly rotated image.
        """
        return util.random_rotation(img, self.degrees, self.resample, self.expand, self.center, self.fill_value)


class RandomSharpness(py_transforms.PyTensorOperation):
    """
    Adjust the sharpness of the input PIL Image by a random degree.

    Args:
        degrees (Sequence[float, float], optional): Range of sharpness adjustment degree to select from, arranged
            in order of (min, max). A degree of ``0.0`` gives a blurred image, a degree of ``1.0``
            gives the original image and a degree of ``2.0`` gives a sharpened image.
            Default: ``(0.1, 1.9)``.

    Raises:
        TypeError : If `degrees` is not of type Sequence[float, float].
        ValueError: If `degrees` is negative.
        ValueError: If `degrees` is not in order of (min, max).

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomSharpness((0.5, 1.5)),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
    @check_positive_degrees
    def __init__(self, degrees=(0.1, 1.9)):
        self.degrees = degrees

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL.Image.Image): Image to be sharpness adjusted.

        Returns:
            PIL.Image.Image, sharpness adjusted image.
        """

        return util.random_sharpness(img, self.degrees)


class RandomVerticalFlip(py_transforms.PyTensorOperation):
    """
    Randomly flip the input PIL Image vertically with a given probability.

    Args:
        prob (float, optional): Probability of performing vertically flip. Default: ``0.5``.

    Raises:
        TypeError: If `prob` is not of type float.
        ValueError: If `prob` is not in range of [0, 1].
        RuntimeError: If shape of input image is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomVerticalFlip(0.5),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
    @check_prob
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL.Image.Image): Image to be vertically flipped.

        Returns:
            PIL.Image.Image, randomly vertically flipped image.
        """
        return util.random_vertical_flip(img, self.prob)


class Resize(py_transforms.PyTensorOperation):
    """
    Resize the input PIL Image to the given size.

    Args:
        size (Union[int, Sequence[int, int]]): The size of the resized image.
            If int is provided, resize the smaller edge of the image to this
            value, keeping the image aspect ratio the same.
            If Sequence[int, int] is provided, its two elements will be taken as the resized height and width.
        interpolation (Inter, optional): Method of interpolation. It can be ``Inter.NEAREST``,
            ``Inter.ANTIALIAS``, ``Inter.BILINEAR`` or ``Inter.BICUBIC``. Default: ``Inter.BILINEAR``.

            - ``Inter.NEAREST`` , nearest-neighbor interpolation.
            - ``Inter.ANTIALIAS`` , antialias interpolation.
            - ``Inter.BILINEA`` , bilinear interpolation.
            - ``Inter.BICUBIC`` , bicubic interpolation.

    Raises:
        TypeError: If `size` is not of type int or Sequence[int, int].
        TypeError: If `interpolation` is not of type :class:`mindspore.dataset.vision.Inter` .
        ValueError: If `size` is not positive.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.Resize(256),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
    @check_resize_interpolation
    def __init__(self, size, interpolation=Inter.BILINEAR):
        self.size = size
        self.interpolation = DE_PY_INTER_MODE.get(interpolation)
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL.Image.Image): Image to be resized.

        Returns:
            PIL.Image.Image, resized image.
        """
        return util.resize(img, self.size, self.interpolation)


class RgbToBgr(py_transforms.PyTensorOperation):
    """
    Convert the input numpy.ndarray images from RGB to BGR.

    Args:
        is_hwc (bool): If ``True``, means the input image is in shape of (H, W, C) or (N, H, W, C).
            Otherwise, it is in shape of (C, H, W) or (N, C, H, W). Default: ``False``.

    Raises:
        TypeError: If `is_hwc` is not of type bool.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.CenterCrop(20),
        ...                            py_vision.ToTensor(),
        ...                            py_vision.RgbToBgr()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision("ConvertColor")
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
    Convert the input numpy.ndarray images from RGB to HSV.

    Args:
        is_hwc (bool): If ``True``, means the input image is in shape of (H, W, C) or (N, H, W, C).
            Otherwise, it is in shape of (C, H, W) or (N, C, H, W). Default: ``False``.

    Raises:
        TypeError: If `is_hwc` is not of type bool.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.CenterCrop(20),
        ...                            py_vision.ToTensor(),
        ...                            py_vision.RgbToHsv()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
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


class TenCrop(py_transforms.PyTensorOperation):
    """
    Crop the given image into one central crop and four corners with the flipped version of these.

    Args:
        size (Union[int, Sequence[int, int]]): The size of the cropped image.
            If int is provided, a square of size `(size, size)` will be cropped with this value.
            If Sequence[int, int] is provided, its two elements will be taken as the cropped height and width.
        use_vertical_flip (bool, optional): If ``True``, flip the images vertically. Otherwise, flip them
            horizontally. Default: ``False``.

    Raises:
        TypeError: If `size` is not of type int or Sequence[int, int].
        TypeError: If `use_vertical_flip` is not of type bool.
        ValueError: If `size` is not positive.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.TenCrop(size=200),
        ...                            # 4D stack of 10 images
        ...                            lambda *images: numpy.stack([py_vision.ToTensor()(image) for image in images])])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
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
            img (PIL.Image.Image): Image to be cropped.

        Returns:
            tuple, 10 cropped PIL.Image.Image, in order of top_left, top_right, bottom_left, bottom_right, center
                of the original image and top_left, top_right, bottom_left, bottom_right, center of the flipped image.
        """
        return util.ten_crop(img, self.size, self.use_vertical_flip)


class ToPIL(py_transforms.PyTensorOperation):
    """
    Convert the input decoded numpy.ndarray image to PIL Image.

    Note:
        The conversion mode will be determined by the data type using `PIL.Image.fromarray` .

    Raises:
        TypeError: If the input image is not of type :class:`numpy.ndarray` or `PIL.Image.Image` .

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> # data is already decoded, but not in PIL Image format
        >>> transforms_list = Compose([py_vision.ToPIL(),
        ...                            py_vision.RandomHorizontalFlip(0.5),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
    def __init__(self):
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (numpy.ndarray): Decoded numpy.ndarray image to be converted to PIL.Image.Image.

        Returns:
            PIL.Image.Image, converted PIL Image.
        """
        return util.to_pil(img)


class ToTensor(py_transforms.PyTensorOperation):
    """
    Convert the input PIL Image or numpy.ndarray to numpy.ndarray of the desired dtype. At the same time,
    the range of pixel value will be changed from [0, 255] to [0.0, 1.0] and the shape will be changed
    from (H, W, C) to (C, H, W).

    Args:
        output_type (numpy.dtype, optional): The desired dtype of the output image. Default: ``numpy.float32`` .

    Raises:
        TypeError: If the input image is not of type `PIL.Image.Image` or :class:`numpy.ndarray` .
        TypeError: If dimension of the input image is not 2 or 3.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> # create a list of transformations to be applied to the "image" column of each data row
        >>> transforms_list = Compose([py_vision.Decode(),
        ...                            py_vision.RandomHorizontalFlip(0.5),
        ...                            py_vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision()
    def __init__(self, output_type=np.float32):
        self.output_type = output_type
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (Union[PIL.Image.Image, numpy.ndarray]): PIL.Image.Image or numpy.ndarray to be type converted.

        Returns:
            numpy.ndarray, converted numpy.ndarray with desired type.
        """
        return util.to_tensor(img, self.output_type)


class ToType(py_transforms.PyTensorOperation):
    """
    Convert the input numpy.ndarray image to the desired dtype.

    Args:
        output_type (numpy.dtype): The desired dtype of the output image, e.g. ``numpy.float32`` .

    Raises:
        TypeError: If the input image is not of type :class:`numpy.ndarray` .

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
        >>> transforms_list =Compose([py_vision.Decode(),
        ...                           py_vision.RandomHorizontalFlip(0.5),
        ...                           py_vision.ToTensor(),
        ...                           py_vision.ToType(np.float32)])
        >>> # apply the transform to dataset through map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns="image")
    """

    @deprecated_py_vision("TypeCast", "mindspore.dataset.transforms")
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


class UniformAugment(py_transforms.PyTensorOperation):
    """
    Uniformly select a number of transformations from a sequence and apply them
    sequentially and randomly, which means that there is a chance that a chosen
    transformation will not be applied.

    All transformations in the sequence require the output type to be the same as
    the input. Thus, the latter one can deal with the output of the previous one.

    Args:
         transforms (Sequence): Sequence of transformations to select from.
         num_ops (int, optional): Number of transformations to be sequentially and randomly applied.
            Default: ``2``.

    Raises:
        TypeError: If `transforms` is not a sequence of data processing operations.
        TypeError: If `num_ops` is not of type int.
        ValueError: If `num_ops` is not positive.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>>
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

    @deprecated_py_vision()
    @check_uniform_augment_py
    def __init__(self, transforms, num_ops=2):
        self.transforms = transforms
        self.num_ops = num_ops
        self.random = False

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL.Image.Image): Image to be transformed.

        Returns:
            PIL.Image.Image, transformed image.
        """
        return util.uniform_augment(img, self.transforms.copy(), self.num_ops)


def not_random(func):
    """
    Specify the function as "not random", i.e., it produces deterministic result.
    A Python function can only be cached after it is specified as "not random".
    """
    func.random = False
    return func
