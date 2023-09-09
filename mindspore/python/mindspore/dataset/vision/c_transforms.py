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
from .utils import Inter, Border, ImageBatchFormat, ConvertMode, SliceMode, AutoAugmentPolicy, parse_padding
from .validators import check_adjust_gamma, check_alpha, check_auto_augment, check_auto_contrast, \
    check_bounding_box_augment_cpp, check_center_crop, check_convert_color, check_crop, check_cut_mix_batch_c, \
    check_cutout, check_gaussian_blur, check_mix_up_batch_c, check_normalize_c, check_normalizepad_c, check_pad, \
    check_positive_degrees, check_prob, check_random_adjust_sharpness, check_random_affine, \
    check_random_auto_contrast, check_random_color_adjust, check_random_crop, check_random_posterize, \
    check_random_resize_crop, check_random_rotation, check_random_select_subpolicy_op, check_random_solarize, \
    check_range, check_rescale, check_resize, check_resize_interpolation, check_rotate, check_slice_patches, \
    check_uniform_augment_cpp, deprecated_c_vision, FLOAT_MAX_INTEGER



class ImageTensorOperation:
    """
    Base class of Image Tensor Ops
    """

    def __call__(self, *input_tensor_list):
        for tensor in input_tensor_list:
            if not isinstance(tensor, (np.ndarray, Image.Image)):
                raise TypeError(
                    "Input should be NumPy or PIL image, got {}.".format(type(tensor)))
        tensor_row = []
        for tensor in input_tensor_list:
            try:
                tensor_row.append(cde.Tensor(np.asarray(tensor)))
            except RuntimeError:
                raise TypeError("Invalid user input. Got {}: {}, cannot be converted into tensor." \
                                .format(type(tensor), tensor))
        callable_op = cde.Execute(self.parse())
        output_tensor_list = callable_op(tensor_row)
        for i, element in enumerate(output_tensor_list):
            arr = element.as_array()
            if arr.dtype.char == 'S':
                output_tensor_list[i] = np.char.decode(arr)
            else:
                output_tensor_list[i] = arr
        return output_tensor_list[0] if len(output_tensor_list) == 1 else tuple(output_tensor_list)

    def parse(self):
        # Note: subclasses must implement `def parse(self)` so do not make ImageTensorOperation's parse a staticmethod.
        raise NotImplementedError("ImageTensorOperation has to implement parse() method.")


DE_C_AUTO_AUGMENT_POLICY = {AutoAugmentPolicy.IMAGENET: cde.AutoAugmentPolicy.DE_AUTO_AUGMENT_POLICY_IMAGENET,
                            AutoAugmentPolicy.CIFAR10: cde.AutoAugmentPolicy.DE_AUTO_AUGMENT_POLICY_CIFAR10,
                            AutoAugmentPolicy.SVHN: cde.AutoAugmentPolicy.DE_AUTO_AUGMENT_POLICY_SVHN}

DE_C_BORDER_TYPE = {Border.CONSTANT: cde.BorderType.DE_BORDER_CONSTANT,
                    Border.EDGE: cde.BorderType.DE_BORDER_EDGE,
                    Border.REFLECT: cde.BorderType.DE_BORDER_REFLECT,
                    Border.SYMMETRIC: cde.BorderType.DE_BORDER_SYMMETRIC}

DE_C_IMAGE_BATCH_FORMAT = {ImageBatchFormat.NHWC: cde.ImageBatchFormat.DE_IMAGE_BATCH_FORMAT_NHWC,
                           ImageBatchFormat.NCHW: cde.ImageBatchFormat.DE_IMAGE_BATCH_FORMAT_NCHW}

DE_C_INTER_MODE = {Inter.NEAREST: cde.InterpolationMode.DE_INTER_NEAREST_NEIGHBOUR,
                   Inter.LINEAR: cde.InterpolationMode.DE_INTER_LINEAR,
                   Inter.CUBIC: cde.InterpolationMode.DE_INTER_CUBIC,
                   Inter.AREA: cde.InterpolationMode.DE_INTER_AREA,
                   Inter.PILCUBIC: cde.InterpolationMode.DE_INTER_PILCUBIC}

DE_C_SLICE_MODE = {SliceMode.PAD: cde.SliceMode.DE_SLICE_PAD,
                   SliceMode.DROP: cde.SliceMode.DE_SLICE_DROP}

DE_C_CONVERT_COLOR_MODE = {ConvertMode.COLOR_BGR2BGRA: cde.ConvertMode.DE_COLOR_BGR2BGRA,
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


class AdjustGamma(ImageTensorOperation):
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
        gain (float, optional): The constant multiplier. Default: ``1.0``.

    Raises:
        TypeError: If `gain` is not of type float.
        TypeError: If `gamma` is not of type float.
        ValueError: If `gamma` is less than 0.
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.AdjustGamma(gamma=10.0, gain=1.0)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    @check_adjust_gamma
    def __init__(self, gamma, gain=1):
        self.gamma = gamma
        self.gain = gain

    def parse(self):
        return cde.AdjustGammaOperation(self.gamma, self.gain)


class AutoAugment(ImageTensorOperation):
    """
    Apply AutoAugment data augmentation method based on
    `AutoAugment: Learning Augmentation Strategies from Data <https://arxiv.org/pdf/1805.09501.pdf>`_ .
    This operation works only with 3-channel RGB images.

    Args:
        policy (AutoAugmentPolicy, optional): AutoAugment policies learned on different datasets.
            Default: ``AutoAugmentPolicy.IMAGENET``.
            It can be any of [AutoAugmentPolicy.IMAGENET, AutoAugmentPolicy.CIFAR10, AutoAugmentPolicy.SVHN].
            Randomly apply 2 operations from a candidate set. See auto augmentation details in AutoAugmentPolicy.

            - AutoAugmentPolicy.IMAGENET, means to apply AutoAugment learned on ImageNet dataset.

            - AutoAugmentPolicy.CIFAR10, means to apply AutoAugment learned on Cifar10 dataset.

            - AutoAugmentPolicy.SVHN, means to apply AutoAugment learned on SVHN dataset.

        interpolation (Inter, optional): Image interpolation mode for Resize operation. Default: ``Inter.NEAREST``.
            It can be ``Inter.NEAREST``, ``Inter.BILINEAR``, ``Inter.BICUBIC``, ``Inter.AREA``.

            - ``Inter.NEAREST`` : means interpolation method is nearest-neighbor interpolation.

            - ``Inter.BILINEA`` : means interpolation method is bilinear interpolation.

            - ``Inter.BICUBIC`` : means the interpolation method is bicubic interpolation.

            - ``Inter.AREA`` : means the interpolation method is pixel area interpolation.

        fill_value (Union[int, tuple], optional): Pixel fill value for the area outside the transformed image.
            It can be an int or a 3-tuple. If it is a 3-tuple, it is used to fill R, G, B channels respectively.
            If it is an integer, it is used for all RGB channels. The `fill_value` values must be in range [0, 255].
            Default: ``0``.

    Raises:
        TypeError: If `policy` is not of type AutoAugmentPolicy.
        TypeError: If `interpolation` is not of type Inter.
        TypeError: If `fill_value` is not an integer or a tuple of length 3.
        RuntimeError: If given tensor shape is not <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.vision import AutoAugmentPolicy, Inter
        >>> transforms_list = [c_vision.Decode(), c_vision.AutoAugment(policy=AutoAugmentPolicy.IMAGENET,
        ...                                                            interpolation=Inter.NEAREST,
        ...                                                            fill_value=0)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    @check_auto_augment
    def __init__(self, policy=AutoAugmentPolicy.IMAGENET, interpolation=Inter.NEAREST, fill_value=0):
        self.policy = policy
        self.interpolation = interpolation
        if isinstance(fill_value, int):
            fill_value = tuple([fill_value] * 3)
        self.fill_value = fill_value

    def parse(self):
        return cde.AutoAugmentOperation(DE_C_AUTO_AUGMENT_POLICY.get(self.policy),
                                        DE_C_INTER_MODE.get(self.interpolation),
                                        self.fill_value)


class AutoContrast(ImageTensorOperation):
    """
    Apply automatic contrast on input image. This operation calculates histogram of image, reassign cutoff percent
    of the lightest pixels from histogram to 255, and reassign cutoff percent of the darkest pixels from histogram to 0.

    Args:
        cutoff (float, optional): Percent of lightest and darkest pixels to cut off from
            the histogram of input image. The value must be in the range [0.0, 50.0). Default: ``0.0``.
        ignore (Union[int, sequence], optional): The background pixel values to ignore,
            The ignore values must be in range [0, 255]. Default: ``None``.

    Raises:
        TypeError: If `cutoff` is not of type float.
        TypeError: If `ignore` is not of type int or sequence.
        ValueError: If `cutoff` is not in range [0, 50.0).
        ValueError: If `ignore` is not in range [0, 255].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.AutoContrast(cutoff=10.0, ignore=[10, 20])]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
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
    Apply a given image processing operation on a random selection of bounding box regions of a given image.

    Args:
        transform (TensorOperation): C++ transformation operation to be applied on random selection
            of bounding box regions of a given image.
        ratio (float, optional): Ratio of bounding boxes to apply augmentation on.
            Range: [0.0, 1.0]. Default: ``0.3``.

    Raises:
        TypeError: If `transform` is not an image processing operation
            in :class:`mindspore.dataset.vision.c_transforms` .
        TypeError: If `ratio` is not of type float.
        ValueError: If `ratio` is not in range [0.0, 1.0].
        RuntimeError: If given bounding box is invalid.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> # set bounding box operation with ratio of 1 to apply rotation on all bounding boxes
        >>> bbox_aug_op = c_vision.BoundingBoxAugment(c_vision.RandomRotation(90), 1)
        >>> # map to apply ops
        >>> image_folder_dataset = image_folder_dataset.map(operations=[bbox_aug_op],
        ...                                                 input_columns=["image", "bbox"],
        ...                                                 output_columns=["image", "bbox"])
    """

    @deprecated_c_vision()
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
    Crop the input image at the center to the given size. If input image size is smaller than output size,
    input image will be padded with 0 before cropping.

    Args:
        size (Union[int, sequence]): The output size of the cropped image.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, an image of size (height, width) will be cropped.
            The size value(s) must be larger than 0.

    Raises:
        TypeError: If `size` is not of type int or sequence.
        ValueError: If `size` is less than or equal to 0.
        RuntimeError: If given tensor shape is not <H, W> or <..., H, W, C>.

    Supported Platforms:
        ``CPU``

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

    @deprecated_c_vision()
    @check_center_crop
    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def parse(self):
        return cde.CenterCropOperation(self.size)


class ConvertColor(ImageTensorOperation):
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
        TypeError: If `convert_mode` is not of type :class:`mindspore.dataset.vision.c_transforms.ConvertMode` .
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import mindspore.dataset.vision.utils as mode
        >>> # Convert RGB images to GRAY images
        >>> convert_op = c_vision.ConvertColor(mode.ConvertMode.COLOR_RGB2GRAY)
        >>> image_folder_dataset = image_folder_dataset.map(operations=convert_op,
        ...                                                 input_columns=["image"])
        >>> # Convert RGB images to BGR images
        >>> convert_op = c_vision.ConvertColor(mode.ConvertMode.COLOR_RGB2BGR)
        >>> image_folder_dataset_1 = image_folder_dataset_1.map(operations=convert_op,
        ...                                                     input_columns=["image"])
    """

    @deprecated_c_vision()
    @check_convert_color
    def __init__(self, convert_mode):
        self.convert_mode = convert_mode

    def parse(self):
        return cde.ConvertColorOperation(DE_C_CONVERT_COLOR_MODE.get(self.convert_mode))


class Crop(ImageTensorOperation):
    """
    Crop the input image at a specific location.

    Args:
        coordinates(sequence): Coordinates of the upper left corner of the cropping image. Must be a sequence of two
            values, in the form of (top, left).
        size (Union[int, sequence]): The output size of the cropped image.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, an image of size (height, width) will be cropped.
            The size value(s) must be larger than 0.

    Raises:
        TypeError: If `coordinates` is not of type sequence.
        TypeError: If `size` is not of type int or sequence.
        ValueError: If `coordinates` is less than 0.
        ValueError: If `size` is less than or equal to 0.
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> decode_op = c_vision.Decode()
        >>> crop_op = c_vision.Crop((0, 0), 32)
        >>> transforms_list = [decode_op, crop_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    @check_crop
    def __init__(self, coordinates, size):
        if isinstance(size, int):
            size = (size, size)
        self.coordinates = coordinates
        self.size = size

    def parse(self):
        return cde.CropOperation(self.coordinates, self.size)


class CutMixBatch(ImageTensorOperation):
    """
    Apply CutMix transformation on input batch of images and labels.
    Note that you need to make labels into one-hot format and batched before calling this operation.

    Args:
        image_batch_format (ImageBatchFormat): The method of padding. Can be
            ``ImageBatchFormat.NHWC``, ``ImageBatchFormat.NCHW``.
        alpha (float, optional): Hyperparameter of beta distribution, must be larger than 0. Default: ``1.0``.
        prob (float, optional): The probability by which CutMix is applied to each image, range: [0, 1].
            Default: ``1.0``.

    Raises:
        TypeError: If `image_batch_format` is not of type :class:`mindspore.dataset.vision.ImageBatchFormat` .
        TypeError: If `alpha` is not of type float.
        TypeError: If `prob` is not of type float.
        ValueError: If `alpha` is less than or equal 0.
        ValueError: If `prob` is not in range [0, 1].
        RuntimeError: If given tensor shape is not <H, W, C>.

    Supported Platforms:
        ``CPU``

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

    @deprecated_c_vision()
    @check_cut_mix_batch_c
    def __init__(self, image_batch_format, alpha=1.0, prob=1.0):
        self.image_batch_format = image_batch_format.value
        self.alpha = alpha
        self.prob = prob

    def parse(self):
        return cde.CutMixBatchOperation(DE_C_IMAGE_BATCH_FORMAT.get(self.image_batch_format), self.alpha, self.prob)


class CutOut(ImageTensorOperation):
    """
    Randomly cut (mask) out a given number of square patches from the input image array.

    Args:
        length (int): The side length of each square patch, must be larger than 0.
        num_patches (int, optional): Number of patches to be cut out of an image, must be larger than 0.
            Default: ``1``.

    Raises:
        TypeError: If `length` is not of type int.
        TypeError: If `num_patches` is not of type int.
        ValueError: If `length` is less than or equal 0.
        ValueError: If `num_patches` is less than or equal 0.
        RuntimeError: If given tensor shape is not <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.CutOut(80, num_patches=10)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    @check_cutout
    def __init__(self, length, num_patches=1):
        self.length = length
        self.num_patches = num_patches

    def parse(self):
        return cde.CutOutOperation(self.length, self.num_patches, True)


class Decode(ImageTensorOperation):
    """
    Decode the input image.

    Args:
        rgb (bool, optional): Mode of decoding input image. Default: ``True``.
            If ``True`` means format of decoded image is RGB else BGR (deprecated).

    Raises:
        RuntimeError: If `rgb` is ``False``, since this option is deprecated.
        RuntimeError: If given tensor is not a 1D sequence.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomHorizontalFlip()]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
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
        if isinstance(img, bytes):
            img = np.frombuffer(img, np.uint8)
        elif not isinstance(img, np.ndarray) or img.ndim != 1 or img.dtype.type is np.str_:
            raise TypeError(
                "Input should be an encoded image in 1-D NumPy format, got {}.".format(type(img)))
        return super().__call__(img)

    def parse(self):
        # deprecated api just support cpu device target
        return cde.DecodeOperation(self.rgb, "CPU")


class Equalize(ImageTensorOperation):
    """
    Apply histogram equalization on input image.

    Raises:
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.Equalize()]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    def __init__(self):
        super().__init__()

    def parse(self):
        return cde.EqualizeOperation()


class GaussianBlur(ImageTensorOperation):
    """
    Blur input image with the specified Gaussian kernel.

    Args:
        kernel_size (Union[int, Sequence[int]]): Size of the Gaussian kernel to use. The value must be positive and odd.
            If only an integer is provided, the kernel size will be (kernel_size, kernel_size). If a sequence of integer
            is provided, it must be a sequence of 2 values which represents (width, height).
        sigma (Union[float, Sequence[float]], optional): Standard deviation of the Gaussian kernel to use.
            Default: ``None``. The value must be positive. If only a float is provided,
            the sigma will be (sigma, sigma).
            If a sequence of float is provided, it must be a sequence of 2 values which represents (width, height).
            If ``None`` is provided, the sigma will be calculated as ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8.

    Raises:
        TypeError: If `kernel_size` is not of type int or Sequence[int].
        TypeError: If `sigma` is not of type float or Sequence[float].
        ValueError: If `kernel_size` is not positive and odd.
        ValueError: If `sigma` is not positive.
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.GaussianBlur(3, 3)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    @check_gaussian_blur
    def __init__(self, kernel_size, sigma=None):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)
        if sigma is None:
            sigma = (0,)
        elif isinstance(sigma, (int, float)):
            sigma = (float(sigma),)
        self.kernel_size = kernel_size
        self.sigma = sigma

    def parse(self):
        return cde.GaussianBlurOperation(self.kernel_size, self.sigma)


class HorizontalFlip(ImageTensorOperation):
    """
    Flip the input image horizontally.

    Raises:
        RuntimeError: If given tensor shape is not <H, W> or <..., H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.HorizontalFlip()]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    def __init__(self):
        super().__init__()

    def parse(self):
        return cde.HorizontalFlipOperation()


class HWC2CHW(ImageTensorOperation):
    """
    Transpose the input image from shape (H, W, C) to (C, H, W).
    If the input image is of shape <H, W>, it will remain unchanged.

    Note:
        This operation supports running on Ascend or GPU platforms by Offload.

    Raises:
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> transforms_list = [c_vision.Decode(),
        ...                    c_vision.RandomHorizontalFlip(0.75),
        ...                    c_vision.RandomCrop(512),
        ...                    c_vision.HWC2CHW()]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    def __init__(self):
        super().__init__()

    def parse(self):
        return cde.HwcToChwOperation()


class Invert(ImageTensorOperation):
    """
    Apply invert on input image in RGB mode. This operation will reassign every pixel to (255 - pixel).

    Raises:
        RuntimeError: If given tensor shape is not <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.Invert()]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    def __init__(self):
        super().__init__()

    def parse(self):
        return cde.InvertOperation()


class MixUpBatch(ImageTensorOperation):
    """
    Apply MixUp transformation on input batch of images and labels. Each image is
    multiplied by a random weight (lambda) and then added to a randomly selected image from the batch
    multiplied by (1 - lambda). The same formula is also applied to the one-hot labels.

    The lambda is generated based on the specified alpha value. Two coefficients x1, x2 are randomly generated
    in the range [alpha, 1], and lambda = (x1 / (x1 + x2)).

    Note that you need to make labels into one-hot format and batched before calling this operation.

    Args:
        alpha (float, optional): Hyperparameter of beta distribution. The value must be positive. Default: ``1.0``.

    Raises:
        TypeError: If `alpha` is not of type float.
        ValueError: If `alpha` is not positive.
        RuntimeError: If given tensor shape is not <N, H, W, C> or <N, C, H, W>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> onehot_op = c_transforms.OneHot(num_classes=10)
        >>> image_folder_dataset= image_folder_dataset.map(operations=onehot_op,
        ...                                                input_columns=["label"])
        >>> mixup_batch_op = c_vision.MixUpBatch(alpha=0.9)
        >>> image_folder_dataset = image_folder_dataset.batch(5)
        >>> image_folder_dataset = image_folder_dataset.map(operations=mixup_batch_op,
        ...                                                 input_columns=["image", "label"])
    """

    @deprecated_c_vision()
    @check_mix_up_batch_c
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def parse(self):
        return cde.MixUpBatchOperation(self.alpha)


class Normalize(ImageTensorOperation):
    """
    Normalize the input image with respect to mean and standard deviation. This operation will normalize
    the input image with: output[channel] = (input[channel] - mean[channel]) / std[channel], where channel >= 1.

    Note:
        This operation supports running on Ascend or GPU platforms by Offload.

    Args:
        mean (sequence): List or tuple of mean values for each channel, with respect to channel order.
            The mean values must be in range [0.0, 255.0].
        std (sequence): List or tuple of standard deviations for each channel, with respect to channel order.
            The standard deviation values must be in range (0.0, 255.0].

    Raises:
        TypeError: If `mean` is not of type sequence.
        TypeError: If `std` is not of type sequence.
        ValueError: If `mean` is not in range [0.0, 255.0].
        ValueError: If `std` is not in range (0.0, 255.0].
        RuntimeError: If given tensor shape is not <H, W> or <...,H, W, C>.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> decode_op = c_vision.Decode()
        >>> normalize_op = c_vision.Normalize(mean=[121.0, 115.0, 100.0], std=[70.0, 68.0, 71.0])
        >>> transforms_list = [decode_op, normalize_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    @check_normalize_c
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def parse(self):
        # deprecated api just support cpu device target
        return cde.NormalizeOperation(self.mean, self.std, True, "CPU")


class NormalizePad(ImageTensorOperation):
    """
    Normalize the input image with respect to mean and standard deviation then pad an extra channel with value zero.

    Args:
        mean (sequence): List or tuple of mean values for each channel, with respect to channel order.
            The mean values must be in range (0.0, 255.0].
        std (sequence): List or tuple of standard deviations for each channel, with respect to channel order.
            The standard deviation values must be in range (0.0, 255.0].
        dtype (str, optional): Set the dtype of the output image. Default: ``"float32"``.

    Raises:
        TypeError: If `mean` is not of type sequence.
        TypeError: If `std` is not of type sequence.
        TypeError: If `dtype` is not of type str.
        ValueError: If `mean` is not in range [0.0, 255.0].
        ValueError: If `std` is not in range (0.0, 255.0].
        RuntimeError: If given tensor shape is not <H, W>, <H, W, C> or <C, H, W>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> decode_op = c_vision.Decode()
        >>> normalize_pad_op = c_vision.NormalizePad(mean=[121.0, 115.0, 100.0],
        ...                                          std=[70.0, 68.0, 71.0],
        ...                                          dtype="float32")
        >>> transforms_list = [decode_op, normalize_pad_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    @check_normalizepad_c
    def __init__(self, mean, std, dtype="float32"):
        self.mean = mean
        self.std = std
        self.dtype = dtype

    def parse(self):
        return cde.NormalizePadOperation(self.mean, self.std, self.dtype, True)


class Pad(ImageTensorOperation):
    """
    Pad the image.

    Args:
        padding (Union[int, Sequence[tuple]]): The number of pixels to pad each border of the image.
            If a single number is provided, it pads all borders with this value.
            If a tuple or lists of 2 values are provided, it pads the (left and top)
            with the first value and (right and bottom) with the second value.
            If 4 values are provided as a list or tuple, it pads the left, top, right and bottom respectively.
            The pad values must be non-negative.
        fill_value (Union[int, tuple[int]], optional): The pixel intensity of the borders, only valid for
            `padding_mode` ``Border.CONSTANT``. If it is a 3-tuple, it is used to fill R, G, B
            channels respectively. If it is an integer, it is used for all RGB channels.
            The `fill_value` values must be in range [0, 255]. Default: ``0``.
        padding_mode (Border, optional): The method of padding. Default: ``Border.CONSTANT``. Can be
            ``Border.CONSTANT``, ``Border.EDGE``, ``Border.REFLECT``, ``Border.SYMMETRIC``.

            - ``Border.CONSTANT`` , means it fills the border with constant values.

            - ``Border.EDGE`` , means it pads with the last value on the edge.

            - ``Border.REFLECT`` , means it reflects the values on the edge omitting the last
              value of edge.

            - ``Border.SYMMETRIC`` , means it reflects the values on the edge repeating the last
              value of edge.

    Note:
        The behavior when `padding` is a sequence of length 2 will change from padding left/top with
        the first value and right/bottom with the second, to padding left/right with the first one
        and top/bottom with the second in the future. Or you can pass in a 4-element sequence to specify
        left, top, right and bottom respectively.

    Raises:
        TypeError: If `padding` is not of type int or Sequence[int].
        TypeError: If `fill_value` is not of type int or tuple[int].
        TypeError: If `padding_mode` is not of type :class:`mindspore.dataset.vision.Border` .
        ValueError: If `padding` is negative.
        ValueError: If `fill_value` is not in range [0, 255].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.Pad([100, 100, 100, 100])]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    @check_pad
    def __init__(self, padding, fill_value=0, padding_mode=Border.CONSTANT):
        padding = parse_padding(padding)
        if isinstance(fill_value, int):
            fill_value = tuple([fill_value] * 3)
        self.padding = padding
        self.fill_value = fill_value
        self.padding_mode = padding_mode

    def parse(self):
        return cde.PadOperation(self.padding, self.fill_value, DE_C_BORDER_TYPE.get(self.padding_mode))


class RandomAdjustSharpness(ImageTensorOperation):
    """
    Randomly adjust the sharpness of the input image with a given probability.

    Args:
        degree (float): Sharpness adjustment degree, which must be non negative.
            Degree of ``0.0`` gives a blurred image, degree of ``1.0`` gives the original image,
            and degree of ``2.0`` increases the sharpness by a factor of 2.
        prob (float, optional): Probability of the image being sharpness adjusted, which
            must be in range of [0, 1]. Default: ``0.5``.

    Raises:
        TypeError: If `degree` is not of type float.
        TypeError: If `prob` is not of type float.
        ValueError: If `degree` is negative.
        ValueError: If `prob` is not in range [0, 1].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomAdjustSharpness(2.0, 0.5)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    @check_random_adjust_sharpness
    def __init__(self, degree, prob=0.5):
        self.prob = prob
        self.degree = degree

    def parse(self):
        return cde.RandomAdjustSharpnessOperation(self.degree, self.prob)


class RandomAffine(ImageTensorOperation):
    """
    Apply Random affine transformation to the input image.

    Args:
        degrees (Union[int, float, sequence]): Range of the rotation degrees.
            If `degrees` is a number, the range will be (-degrees, degrees).
            If `degrees` is a sequence, it should be (min, max).
        translate (sequence, optional): Sequence (tx_min, tx_max, ty_min, ty_max) of minimum/maximum translation in
            x(horizontal) and y(vertical) directions, range [-1.0, 1.0]. Default: ``None``.
            The horizontal and vertical shift is selected randomly from the range:
            (tx_min*width, tx_max*width) and (ty_min*height, ty_max*height), respectively.
            If a tuple or list of size 2, then a translate parallel to the X axis in the range of
            (translate[0], translate[1]) is applied.
            If a tuple or list of size 4, then a translate parallel to the X axis in the range of
            (translate[0], translate[1]) and a translate parallel to the Y axis in the range of
            (translate[2], translate[3]) are applied.
            If ``None``, no translation is applied.
        scale (sequence, optional): Scaling factor interval, which must be non negative.
            Default: ``None``, original scale is used.
        shear (Union[int, float, sequence], optional): Range of shear factor, which must be positive.
            Default: ``None``.
            If a number, then a shear parallel to the X axis in the range of (-shear, +shear) is applied.
            If a tuple or list of size 2, then a shear parallel to the X axis in the range of (shear[0], shear[1])
            is applied.
            If a tuple or list of size 4, then a shear parallel to X axis in the range of (shear[0], shear[1])
            and a shear parallel to Y axis in the range of (shear[2], shear[3]) is applied.
            If None, no shear is applied.
        resample (Inter, optional): An optional resampling filter. Default: ``Inter.NEAREST``.
            It can be ``Inter.BILINEAR``, ``Inter.NEAREST``, ``Inter.BICUBIC``, ``Inter.AREA``.

            - ``Inter.BILINEA`` , means resample method is bilinear interpolation.

            - ``Inter.NEAREST`` , means resample method is nearest-neighbor interpolation.

            - ``Inter.BICUBIC`` , means resample method is bicubic interpolation.

            - ``Inter.AREA`` :, means resample method is pixel area interpolation.

        fill_value (Union[int, tuple[int]], optional): Optional fill_value to fill the area outside the transform
            in the output image. There must be three elements in tuple and the value of single element is [0, 255].
            Default: ``0``, filling is performed.

    Raises:
        TypeError: If `degrees` is not of type int, float or sequence.
        TypeError: If `translate` is not of type sequence.
        TypeError: If `scale` is not of type sequence.
        TypeError: If `shear` is not of type int, float or sequence.
        TypeError: If `resample` is not of type :class:`mindspore.dataset.vision.Inter` .
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
        >>> decode_op = c_vision.Decode()
        >>> random_affine_op = c_vision.RandomAffine(degrees=15,
        ...                                          translate=(-0.1, 0.1, 0, 0),
        ...                                          scale=(0.9, 1.1),
        ...                                          resample=Inter.NEAREST)
        >>> transforms_list = [decode_op, random_affine_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
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
        self.resample = DE_C_INTER_MODE.get(resample)
        self.fill_value = fill_value

    def parse(self):
        return cde.RandomAffineOperation(self.degrees, self.translate, self.scale_, self.shear, self.resample,
                                         self.fill_value)


class RandomAutoContrast(ImageTensorOperation):
    """
    Automatically adjust the contrast of the image with a given probability.

    Args:
        cutoff (float, optional): Percent of the lightest and darkest pixels to be cut off from
            the histogram of the input image. The value must be in range of [0.0, 50.0). Default: ``0.0``.
        ignore (Union[int, sequence], optional): The background pixel values to be ignored, each of
            which must be in range of [0, 255]. Default: ``None``.
        prob (float, optional): Probability of the image being automatically contrasted, which
            must be in range of [0, 1]. Default: ``0.5``.

    Raises:
        TypeError: If `cutoff` is not of type float.
        TypeError: If `ignore` is not of type int or sequence of int.
        TypeError: If `prob` is not of type float.
        ValueError: If `cutoff` is not in range [0.0, 50.0).
        ValueError: If `ignore` is not in range [0, 255].
        ValueError: If `prob` is not in range [0, 1].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomAutoContrast(cutoff=0.0, ignore=None, prob=0.5)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    @check_random_auto_contrast
    def __init__(self, cutoff=0.0, ignore=None, prob=0.5):
        if ignore is None:
            ignore = []
        if isinstance(ignore, int):
            ignore = [ignore]
        self.cutoff = cutoff
        self.ignore = ignore
        self.prob = prob

    def parse(self):
        return cde.RandomAutoContrastOperation(self.cutoff, self.ignore, self.prob)


class RandomColor(ImageTensorOperation):
    """
    Adjust the color of the input image by a fixed or random degree.
    This operation works only with 3-channel RGB images.

    Args:
         degrees (Sequence[float], optional): Range of random color adjustment degrees, which must be non-negative.
            It should be in (min, max) format. If min=max, then it is a
            single fixed magnitude operation. Default: ``(0.1, 1.9)``.

    Raises:
        TypeError: If `degrees` is not of type Sequence[float].
        ValueError: If `degrees` is negative.
        RuntimeError: If given tensor shape is not <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomColor((0.5, 2.0))]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    @check_positive_degrees
    def __init__(self, degrees=(0.1, 1.9)):
        self.degrees = degrees

    def parse(self):
        return cde.RandomColorOperation(*self.degrees)


class RandomColorAdjust(ImageTensorOperation):
    """
    Randomly adjust the brightness, contrast, saturation, and hue of the input image.

    Note:
        This operation supports running on Ascend or GPU platforms by Offload.

    Args:
        brightness (Union[float, Sequence[float]], optional): Brightness adjustment factor. Default: ``(1, 1)``.
            Cannot be negative.
            If it is a float, the factor is uniformly chosen from the range [max(0, 1-brightness), 1+brightness].
            If it is a sequence, it should be [min, max] for the range.
        contrast (Union[float, Sequence[float]], optional): Contrast adjustment factor. Default: ``(1, 1)``.
            Cannot be negative.
            If it is a float, the factor is uniformly chosen from the range [max(0, 1-contrast), 1+contrast].
            If it is a sequence, it should be [min, max] for the range.
        saturation (Union[float, Sequence[float]], optional): Saturation adjustment factor. Default: ``(1, 1)``.
            Cannot be negative.
            If it is a float, the factor is uniformly chosen from the range [max(0, 1-saturation), 1+saturation].
            If it is a sequence, it should be [min, max] for the range.
        hue (Union[float, Sequence[float]], optional): Hue adjustment factor. Default: ``(0, 0)``.
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
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> decode_op = c_vision.Decode()
        >>> transform_op = c_vision.RandomColorAdjust(brightness=(0.5, 1),
        ...                                           contrast=(0.4, 1),
        ...                                           saturation=(0.3, 1))
        >>> transforms_list = [decode_op, transform_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    @check_random_color_adjust
    def __init__(self, brightness=(1, 1), contrast=(1, 1), saturation=(1, 1), hue=(0, 0)):
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

    def __expand_values(self, value, center=1, bound=(0, FLOAT_MAX_INTEGER), non_negative=True):
        """Expand input value for vision adjustment factor."""
        if isinstance(value, numbers.Number):
            value = [center - value, center + value]
            if non_negative:
                value[0] = max(0, value[0])
            check_range(value, bound)
        return (value[0], value[1])


class RandomCrop(ImageTensorOperation):
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
            The padding value(s) must be non-negative. Default: ``None``.
            If `padding` is not ``None``, pad image first with padding values.
            If a single number is provided, pad all borders with this value.
            If a tuple or lists of 2 values are provided, pad the (left and top)
            with the first value and (right and bottom) with the second value.
            If 4 values are provided as a list or tuple,
            pad the left, top, right and bottom respectively.
        pad_if_needed (bool, optional): Pad the image if either side is smaller than
            the given output size. Default: ``False``.
        fill_value (Union[int, tuple[int]], optional): The pixel intensity of the borders, only valid for
            `padding_mode` ``Border.CONSTANT``. If it is a 3-tuple, it is used to fill R, G, B channels respectively.
            If it is an integer, it is used for all RGB channels.
            The fill_value values must be in range [0, 255]. Default: ``0``.
        padding_mode (Border, optional): The method of padding. Default: ``Border.CONSTANT``. It can be
            ``Border.CONSTANT``, ``Border.EDGE``, ``Border.REFLECT``, ``Border.SYMMETRIC``.

            - ``Border.CONSTANT`` , means it fills the border with constant values.

            - ``Border.EDGE`` , means it pads with the last value on the edge.

            - ``Border.REFLECT`` , means it reflects the values on the edge omitting the last
              value of edge.

            - ``Border.SYMMETRIC`` , means it reflects the values on the edge repeating the last
              value of edge.

    Note:
        The behavior when `padding` is a sequence of length 2 will change from padding left/top with
        the first value and right/bottom with the second, to padding left/right with the first one
        and top/bottom with the second in the future. Or you can pass in a 4-element sequence to specify
        left, top, right and bottom respectively.

    Raises:
        TypeError: If `size` is not of type int or Sequence[int].
        TypeError: If `padding` is not of type int or Sequence[int].
        TypeError: If `pad_if_needed` is not of type boolean.
        TypeError: If `fill_value` is not of type int or tuple[int].
        TypeError: If `padding_mode` is not of type :class:`mindspore.dataset.vision.Border` .
        ValueError: If `size` is not positive.
        ValueError: If `padding` is negative.
        ValueError: If `fill_value` is not in range [0, 255].
        RuntimeError: If given tensor shape is not <H, W> or <..., H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.vision import Border
        >>> decode_op = c_vision.Decode()
        >>> random_crop_op = c_vision.RandomCrop(512, [200, 200, 200, 200], padding_mode=Border.EDGE)
        >>> transforms_list = [decode_op, random_crop_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
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
        self.padding_mode = padding_mode

    def parse(self):
        border_type = DE_C_BORDER_TYPE.get(self.padding_mode)
        return cde.RandomCropOperation(self.size, self.padding, self.pad_if_needed, self.fill_value, border_type)


class RandomCropDecodeResize(ImageTensorOperation):
    """
    A combination of `Crop` , `Decode` and `Resize` . It will get better performance for JPEG images. This operation
    will crop the input image at a random location, decode the cropped image in RGB mode, and resize the decoded image.

    Args:
        size (Union[int, Sequence[int]]): The output size of the resized image. The size value(s) must be positive.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, an image of size (height, width) will be cropped.
        scale (Union[list, tuple], optional): Range [min, max) of respective size of the
            original size to be cropped, which must be non-negative. Default: ``(0.08, 1.0)``.
        ratio (Union[list, tuple], optional): Range [min, max) of aspect ratio to be
            cropped, which must be non-negative. Default: ``(3. / 4., 4. / 3.)``.
        interpolation (Inter, optional): Image interpolation mode for resize operation. Default: ``Inter.BILINEAR``.
            It can be ``Inter.BILINEAR``, ``Inter.NEAREST``, ``Inter.BICUBIC``, ``Inter.AREA``, ``Inter.PILCUBIC``.

            - ``Inter.BILINEA`` , means interpolation method is bilinear interpolation.

            - ``Inter.NEAREST`` , means interpolation method is nearest-neighbor interpolation.

            - ``Inter.BICUBIC`` , means interpolation method is bicubic interpolation.

            - ``Inter.AREA`` :, means interpolation method is pixel area interpolation.

            - ``Inter.PILCUBIC`` , means interpolation method is bicubic interpolation like implemented in pillow, input
              should be in 3 channels format.

        max_attempts (int, optional): The maximum number of attempts to propose a valid crop_area. Default: ``10``.
            If exceeded, fall back to use center_crop instead. The `max_attempts` value must be positive.

    Raises:
        TypeError: If `size` is not of type int or Sequence[int].
        TypeError: If `scale` is not of type tuple or list.
        TypeError: If `ratio` is not of type tuple or list.
        TypeError: If `interpolation` is not of type :class:`mindspore.dataset.vision.Inter` .
        TypeError: If `max_attempts` is not of type int.
        ValueError: If `size` is not positive.
        ValueError: If `scale` is negative.
        ValueError: If `ratio` is negative.
        ValueError: If `max_attempts` is not positive.
        RuntimeError: If given tensor is not a 1D sequence.

    Supported Platforms:
        ``CPU``

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

    @deprecated_c_vision()
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
                                                   DE_C_INTER_MODE.get(self.interpolation),
                                                   self.max_attempts)


class RandomCropWithBBox(ImageTensorOperation):
    """
    Crop the input image at a random location and adjust bounding boxes accordingly.

    Args:
        size (Union[int, Sequence[int]]): The output size of the cropped image. The size value(s) must be positive.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, an image of size (height, width) will be cropped.
        padding (Union[int, Sequence[int]], optional): The number of pixels to pad the image
            The padding value(s) must be non-negative. Default: ``None``.
            If `padding` is not ``None``, first pad image with padding values.
            If a single number is provided, pad all borders with this value.
            If a tuple or lists of 2 values are provided, pad the (left and top)
            with the first value and (right and bottom) with the second value.
            If 4 values are provided as a list or tuple, pad the left, top, right and bottom respectively.
        pad_if_needed (bool, optional): Pad the image if either side is smaller than
            the given output size. Default: ``False``.
        fill_value (Union[int, tuple[int]], optional): The pixel intensity of the borders, only valid for
            `padding_mode` ``Border.CONSTANT``. If it is a 3-tuple, it is used to fill R, G, B channels respectively.
            If it is an integer, it is used for all RGB channels.
            The fill_value values must be in range [0, 255]. Default: ``0``.
        padding_mode (Border, optional): The method of padding. Default: ``Border.CONSTANT``. It can be any of
            ``Border.CONSTANT``, ``Border.EDGE``, ``Border.REFLECT``, ``Border.SYMMETRIC``.

            - ``Border.CONSTANT`` , means it fills the border with constant values.

            - ``Border.EDGE`` , means it pads with the last value on the edge.

            - ``Border.REFLECT`` , means it reflects the values on the edge omitting the last
              value of edge.

            - ``Border.SYMMETRIC`` , means it reflects the values on the edge repeating the last
              value of edge.

    Note:
        The behavior when `padding` is a sequence of length 2 will change from padding left/top with
        the first value and right/bottom with the second, to padding left/right with the first one
        and top/bottom with the second in the future. Or you can pass in a 4-element sequence to specify
        left, top, right and bottom respectively.

    Raises:
        TypeError: If `size` is not of type int or Sequence[int].
        TypeError: If `padding` is not of type int or Sequence[int].
        TypeError: If `pad_if_needed` is not of type boolean.
        TypeError: If `fill_value` is not of type int or tuple[int].
        TypeError: If `padding_mode` is not of type :class:`mindspore.dataset.vision.Border` .
        ValueError: If `size` is not positive.
        ValueError: If `padding` is negative.
        ValueError: If `fill_value` is not in range [0, 255].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> decode_op = c_vision.Decode()
        >>> random_crop_with_bbox_op = c_vision.RandomCropWithBBox([512, 512], [200, 200, 200, 200])
        >>> transforms_list = [decode_op, random_crop_with_bbox_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
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
        self.padding_mode = padding_mode

    def parse(self):
        border_type = DE_C_BORDER_TYPE.get(self.padding_mode)
        return cde.RandomCropWithBBoxOperation(self.size, self.padding, self.pad_if_needed, self.fill_value,
                                               border_type)


class RandomEqualize(ImageTensorOperation):
    """
    Apply histogram equalization on the input image with a given probability.

    Args:
        prob (float, optional): Probability of the image being equalized, which
            must be in range of [0, 1]. Default: ``0.5``.

    Raises:
        TypeError: If `prob` is not of type float.
        ValueError: If `prob` is not in range [0, 1].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomEqualize(0.5)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    @check_prob
    def __init__(self, prob=0.5):
        self.prob = prob

    def parse(self):
        return cde.RandomEqualizeOperation(self.prob)


class RandomHorizontalFlip(ImageTensorOperation):
    """
    Randomly flip the input image horizontally with a given probability.

    Note:
        This operation supports running on Ascend or GPU platforms by Offload.

    Args:
        prob (float, optional): Probability of the image being flipped, which must be in range of [0, 1].
            Default: ``0.5``.

    Raises:
        TypeError: If `prob` is not of type float.
        ValueError: If `prob` is not in range [0, 1].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomHorizontalFlip(0.75)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    @check_prob
    def __init__(self, prob=0.5):
        self.prob = prob

    def parse(self):
        return cde.RandomHorizontalFlipOperation(self.prob)


class RandomHorizontalFlipWithBBox(ImageTensorOperation):
    """
    Flip the input image horizontally randomly with a given probability and adjust bounding boxes accordingly.

    Args:
        prob (float, optional): Probability of the image being flipped, which must be in range of [0, 1].
            Default: ``0.5``.

    Raises:
        TypeError: If `prob` is not of type float.
        ValueError: If `prob` is not in range [0, 1].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomHorizontalFlipWithBBox(0.70)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    @check_prob
    def __init__(self, prob=0.5):
        self.prob = prob

    def parse(self):
        return cde.RandomHorizontalFlipWithBBoxOperation(self.prob)


class RandomInvert(ImageTensorOperation):
    """
    Randomly invert the colors of image with a given probability.

    Args:
        prob (float, optional): Probability of the image being inverted, which must be in range of [0, 1].
            Default: ``0.5``.

    Raises:
        TypeError: If `prob` is not of type float.
        ValueError: If `prob` is not in range [0, 1].
        RuntimeError: If given tensor shape is not <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomInvert(0.5)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    @check_prob
    def __init__(self, prob=0.5):
        self.prob = prob

    def parse(self):
        return cde.RandomInvertOperation(self.prob)


class RandomLighting(ImageTensorOperation):
    """
    Add AlexNet-style PCA-based noise to an image. The eigenvalue and eigenvectors for Alexnet's PCA noise is
    calculated from the imagenet dataset.

    Args:
        alpha (float, optional): Intensity of the image, which must be non-negative. Default: ``0.05``.

    Raises:
        TypeError: If `alpha` is not of type float.
        ValueError: If `alpha` is negative.
        RuntimeError: If given tensor shape is not <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomLighting(0.1)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    @check_alpha
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def parse(self):
        return cde.RandomLightingOperation(self.alpha)


class RandomPosterize(ImageTensorOperation):
    """
    Reduce the number of bits for each color channel to posterize the input image randomly with a given probability.

    Args:
        bits (Union[int, Sequence[int]], optional): Range of random posterize to compress image.
            Bits values must be in range of [1,8], and include at
            least one integer value in the given range. It must be in
            (min, max) or integer format. If min=max, then it is a single fixed
            magnitude operation. Default: ``(8, 8)``.

    Raises:
        TypeError: If `bits` is not of type int or sequence of int.
        ValueError: If `bits` is not in range [1, 8].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomPosterize((6, 8))]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    @check_random_posterize
    def __init__(self, bits=(8, 8)):
        self.bits = bits

    def parse(self):
        bits = self.bits
        if isinstance(bits, int):
            bits = (bits, bits)
        return cde.RandomPosterizeOperation(bits)


class RandomResizedCrop(ImageTensorOperation):
    """
    This operation will crop the input image randomly, and resize the cropped image using a selected interpolation mode.

    Note:
        If the input image is more than one, then make sure that the image size is the same.

    Args:
        size (Union[int, Sequence[int]]): The output size of the resized image. The size value(s) must be positive.
            If size is an integer, a square of size (size, size) will be cropped with this value.
            If size is a sequence of length 2, an image of size (height, width) will be cropped.
        scale (Union[list, tuple], optional): Range [min, max) of respective size of the original
            size to be cropped, which must be non-negative. Default: ``(0.08, 1.0)``.
        ratio (Union[list, tuple], optional): Range [min, max) of aspect ratio to be
            cropped, which must be non-negative. Default: ``(3. / 4., 4. / 3.)``.
        interpolation (Inter, optional): Method of interpolation. Default: ``Inter.BILINEAR``.
            It can be ``Inter.BILINEAR``, ``Inter.NEAREST``, ``Inter.BICUBIC``, ``Inter.AREA``, ``Inter.PILCUBIC``.

            - ``Inter.BILINEA`` , means interpolation method is bilinear interpolation.

            - ``Inter.NEAREST`` , means interpolation method is nearest-neighbor interpolation.

            - ``Inter.BICUBIC`` , means interpolation method is bicubic interpolation.

            - ``Inter.AREA`` :, means interpolation method is pixel area interpolation.

            - ``Inter.PILCUBIC`` , means interpolation method is bicubic interpolation like implemented in pillow,
              input should be in 3 channels format.

        max_attempts (int, optional): The maximum number of attempts to propose a valid
            crop_area. Default: ``10``. If exceeded, fall back to use center_crop instead.

    Raises:
        TypeError: If `size` is not of type int or Sequence[int].
        TypeError: If `scale` is not of type tuple or list.
        TypeError: If `ratio` is not of type tuple or list.
        TypeError: If `interpolation` is not of type :class:`mindspore.dataset.vision.Inter` .
        TypeError: If `max_attempts` is not of type int.
        ValueError: If `size` is not positive.
        ValueError: If `scale` is negative.
        ValueError: If `ratio` is negative.
        ValueError: If `max_attempts` is not positive.
        RuntimeError: If given tensor shape is not <H, W> or <..., H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.vision import Inter
        >>> decode_op = c_vision.Decode()
        >>> resize_crop_op = c_vision.RandomResizedCrop(size=(50, 75), scale=(0.25, 0.5),
        ...                                             interpolation=Inter.BILINEAR)
        >>> transforms_list = [decode_op, resize_crop_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
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
        return cde.RandomResizedCropOperation(self.size, self.scale, self.ratio,
                                              DE_C_INTER_MODE.get(self.interpolation), self.max_attempts)


class RandomResizedCropWithBBox(ImageTensorOperation):
    """
    Crop the input image to a random size and aspect ratio and adjust bounding boxes accordingly.

    Args:
        size (Union[int, Sequence[int]]): The size of the output image. The size value(s) must be positive.
            If size is an integer, a square of size (size, size) will be cropped with this value.
            If size is a sequence of length 2, an image of size (height, width) will be cropped.
        scale (Union[list, tuple], optional): Range (min, max) of respective size of the original
            size to be cropped, which must be non-negative. Default: ``(0.08, 1.0)``.
        ratio (Union[list, tuple], optional): Range (min, max) of aspect ratio to be
            cropped, which must be non-negative. Default: ``(3. / 4., 4. / 3.)``.
        interpolation (Inter mode, optional): Method of interpolation. Default: ``Inter.BILINEAR``.
            It can be ``Inter.BILINEAR``, ``Inter.NEAREST``, ``Inter.BICUBIC`` .

            - ``Inter.BILINEA`` , means interpolation method is bilinear interpolation.

            - ``Inter.NEAREST`` , means interpolation method is nearest-neighbor interpolation.

            - ``Inter.BICUBIC`` , means interpolation method is bicubic interpolation.

        max_attempts (int, optional): The maximum number of attempts to propose a valid
            crop area. Default: ``10``. If exceeded, fall back to use center crop instead.

    Raises:
        TypeError: If `size` is not of type int or Sequence[int].
        TypeError: If `scale` is not of type tuple or list.
        TypeError: If `ratio` is not of type tuple or list.
        TypeError: If `interpolation` is not of type :class:`mindspore.dataset.vision.Inter` .
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
        >>> decode_op = c_vision.Decode()
        >>> bbox_op = c_vision.RandomResizedCropWithBBox(size=50, interpolation=Inter.NEAREST)
        >>> transforms_list = [decode_op, bbox_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
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
                                                      DE_C_INTER_MODE.get(self.interpolation), self.max_attempts)


class RandomResize(ImageTensorOperation):
    """
    Resize the input image using a randomly selected interpolation mode.

    Args:
        size (Union[int, Sequence[int]]): The output size of the resized image. The size value(s) must be positive.
            If size is an integer, a square of size (size, size) will be cropped with this value.
            If size is a sequence of length 2, an image of size (height, width) will be cropped.

    Raises:
        TypeError: If `size` is not of type int or Sequence[int].
        ValueError: If `size` is not positive.
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

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

    @deprecated_c_vision()
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
        size (Union[int, Sequence[int]]): The output size of the resized image. The size value(s) must be positive.
            If size is an integer, a square of size (size, size) will be cropped with this value.
            If size is a sequence of length 2, an image of size (height, width) will be cropped.

    Raises:
        TypeError: If `size` is not of type int or Sequence[int].
        ValueError: If `size` is not positive.
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

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

    @deprecated_c_vision()
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
    Rotate the input image randomly within a specified range of degrees.

    Args:
        degrees (Union[int, float, sequence]): Range of random rotation degrees.
            If `degrees` is a number, the range will be converted to (-degrees, degrees).
            If `degrees` is a sequence, it should be (min, max).
        resample (Inter, optional): An optional resampling filter. Default: ``Inter.NEAREST``.
            It can be ``Inter.BILINEAR``, ``Inter.NEAREST``, ``Inter.BICUBIC``, ``Inter.AREA``.

            - ``Inter.BILINEA`` , means resample method is bilinear interpolation.

            - ``Inter.NEAREST`` , means resample method is nearest-neighbor interpolation.

            - ``Inter.BICUBIC`` , means resample method is bicubic interpolation.

            - ``Inter.AREA`` : means the interpolation method is pixel area interpolation.

        expand (bool, optional):  Optional expansion flag. Default: ``False``. If set to ``True``, expand the output
            image to make it large enough to hold the entire rotated image.
            If set to ``False`` or omitted, make the output image the same size as the input.
            Note that the expand flag assumes rotation around the center and no translation.
        center (tuple, optional): Optional center of rotation (a 2-tuple). Default: ``None``.
            Origin is the top left corner. ``None`` sets to the center of the image.
        fill_value (Union[int, tuple[int]], optional): Optional fill color for the area outside the rotated image.
            If it is a 3-tuple, it is used to fill R, G, B channels respectively.
            If it is an integer, it is used for all RGB channels.
            The fill_value values must be in range [0, 255]. Default: ``0``.

    Raises:
        TypeError: If `degrees` is not of type int, float or sequence.
        TypeError: If `resample` is not of type :class:`mindspore.dataset.vision.Inter` .
        TypeError: If `expand` is not of type boolean.
        TypeError: If `center` is not of type tuple.
        TypeError: If `fill_value` is not of type int or tuple[int].
        ValueError: If `fill_value` is not in range [0, 255].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.vision import Inter
        >>> transforms_list = [c_vision.Decode(),
        ...                    c_vision.RandomRotation(degrees=5.0,
        ...                    resample=Inter.NEAREST,
        ...                    expand=True)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    @check_random_rotation
    def __init__(self, degrees, resample=Inter.NEAREST, expand=False, center=None, fill_value=0):
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
        if center is None:
            center = ()
        if isinstance(fill_value, int):
            fill_value = tuple([fill_value] * 3)
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill_value = fill_value

    def parse(self):
        return cde.RandomRotationOperation(self.degrees, DE_C_INTER_MODE.get(self.resample), self.expand, self.center,
                                           self.fill_value)


class RandomSelectSubpolicy(ImageTensorOperation):
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
        >>> policy = [[(c_vision.RandomRotation((45, 45)), 0.5),
        ...            (c_vision.RandomVerticalFlip(), 1),
        ...            (c_vision.RandomColorAdjust(), 0.8)],
        ...           [(c_vision.RandomRotation((90, 90)), 1),
        ...            (c_vision.RandomColorAdjust(), 0.2)]]
        >>> image_folder_dataset = image_folder_dataset.map(operations=c_vision.RandomSelectSubpolicy(policy),
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    @check_random_select_subpolicy_op
    def __init__(self, policy):
        self.policy = policy

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


class RandomSharpness(ImageTensorOperation):
    """
    Adjust the sharpness of the input image by a fixed or random degree. Degree of 0.0 gives a blurred image,
    degree of 1.0 gives the original image, and degree of 2.0 gives a sharpened image.

    Note:
        This operation supports running on Ascend or GPU platforms by Offload.

    Args:
        degrees (Union[list, tuple], optional): Range of random sharpness adjustment degrees,
            which must be non-negative. It should be in (min, max) format. If min=max, then
            it is a single fixed magnitude operation. Default: ``(0.1, 1.9)``.

    Raises:
        TypeError : If `degrees` is not of type list or tuple.
        ValueError: If `degrees` is negative.
        ValueError: If `degrees` is in (max, min) format instead of (min, max).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomSharpness(degrees=(0.2, 1.9))]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    @check_positive_degrees
    def __init__(self, degrees=(0.1, 1.9)):
        self.degrees = degrees

    def parse(self):
        return cde.RandomSharpnessOperation(self.degrees)


class RandomSolarize(ImageTensorOperation):
    """
    Randomly selects a subrange within the specified threshold range and sets the pixel value within
    the subrange to (255 - pixel).

    Args:
        threshold (tuple, optional): Range of random solarize threshold. Default: ``(0, 255)``.
            Threshold values should always be in (min, max) format,
            where min and max are integers in the range [0, 255], and min <= max.
            If min=max, then invert all pixel values above min(max).

    Raises:
        TypeError : If `threshold` is not of type tuple.
        ValueError: If `threshold` is not in range of [0, 255].

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomSolarize(threshold=(10,100))]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    @check_random_solarize
    def __init__(self, threshold=(0, 255)):
        self.threshold = threshold

    def parse(self):
        return cde.RandomSolarizeOperation(self.threshold)


class RandomVerticalFlip(ImageTensorOperation):
    """
    Randomly flip the input image vertically with a given probability.

    Note:
        This operation supports running on Ascend or GPU platforms by Offload.

    Args:
        prob (float, optional): Probability of the image being flipped. Default: ``0.5``.

    Raises:
        TypeError: If `prob` is not of type float.
        ValueError: If `prob` is not in range [0, 1].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomVerticalFlip(0.25)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    @check_prob
    def __init__(self, prob=0.5):
        self.prob = prob

    def parse(self):
        return cde.RandomVerticalFlipOperation(self.prob)


class RandomVerticalFlipWithBBox(ImageTensorOperation):
    """
    Flip the input image vertically, randomly with a given probability and adjust bounding boxes accordingly.

    Args:
        prob (float, optional): Probability of the image being flipped. Default: ``0.5``.

    Raises:
        TypeError: If `prob` is not of type float.
        ValueError: If `prob` is not in range [0, 1].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomVerticalFlipWithBBox(0.20)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    @check_prob
    def __init__(self, prob=0.5):
        self.prob = prob

    def parse(self):
        return cde.RandomVerticalFlipWithBBoxOperation(self.prob)


class Rescale(ImageTensorOperation):
    """
    Rescale the input image with the given rescale and shift. This operation will rescale the input image
    with: output = image * rescale + shift.

    Note:
        This operation supports running on Ascend or GPU platforms by Offload.

    Args:
        rescale (float): Rescale factor.
        shift (float): Shift factor.

    Raises:
        TypeError: If `rescale` is not of type float.
        TypeError: If `shift` is not of type float.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.Rescale(1.0 / 255.0, -1.0)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    @check_rescale
    def __init__(self, rescale, shift):
        self.rescale = rescale
        self.shift = shift

    def parse(self):
        return cde.RescaleOperation(self.rescale, self.shift)


class Resize(ImageTensorOperation):
    """
    Resize the input image to the given size with a given interpolation mode.

    Args:
        size (Union[int, Sequence[int]]): The output size of the resized image. The size value(s) must be positive.
            If size is an integer, the smaller edge of the image will be resized to this value with
            the same image aspect ratio.
            If size is a sequence of length 2, it should be (height, width).
        interpolation (Inter, optional): Image interpolation mode. Default: ``Inter.LINEAR``.
            It can be ``Inter.LINEAR``, ``Inter.NEAREST``, ``Inter.BICUBIC``, ``Inter.AREA``, ``Inter.PILCUBIC``.

            - ``Inter.LINEAR`` , means interpolation method is bilinear interpolation.

            - ``Inter.NEAREST`` , means interpolation method is nearest-neighbor interpolation.

            - ``Inter.BICUBIC`` , means interpolation method is bicubic interpolation.

            - ``Inter.AREA`` :, means interpolation method is pixel area interpolation.

            - ``Inter.PILCUBIC`` , means interpolation method is bicubic interpolation like implemented in pillow, input
              should be in 3 channels format.

    Raises:
        TypeError: If `size` is not of type int or Sequence[int].
        TypeError: If `interpolation` is not of type :class:`mindspore.dataset.vision.Inter` .
        ValueError: If `size` is not positive.
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.vision import Inter
        >>> decode_op = c_vision.Decode()
        >>> resize_op = c_vision.Resize([100, 75], Inter.BICUBIC)
        >>> transforms_list = [decode_op, resize_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    @check_resize_interpolation
    def __init__(self, size, interpolation=Inter.LINEAR):
        if isinstance(size, int):
            size = (size,)
        self.size = size
        self.interpolation = interpolation

    def parse(self):
        # deprecated api just support cpu device target
        return cde.ResizeOperation(self.size, DE_C_INTER_MODE.get(self.interpolation), "CPU")


class ResizeWithBBox(ImageTensorOperation):
    """
    Resize the input image to the given size and adjust bounding boxes accordingly.

    Args:
        size (Union[int, Sequence[int]]): The output size of the resized image.
            If size is an integer, smaller edge of the image will be resized to this value with
            the same image aspect ratio.
            If size is a sequence of length 2, it should be (height, width).
        interpolation (Inter, optional): Image interpolation mode. Default: ``Inter.LINEAR``.
            It can be ``Inter.LINEAR``, ``Inter.NEAREST``, ``Inter.BICUBIC``.

            - ``Inter.LINEAR`` , means interpolation method is bilinear interpolation.

            - ``Inter.NEAREST`` , means interpolation method is nearest-neighbor interpolation.

            - ``Inter.BICUBIC`` , means interpolation method is bicubic interpolation.

    Raises:
        TypeError: If `size` is not of type int or Sequence[int].
        TypeError: If `interpolation` is not of type :class:`mindspore.dataset.vision.Inter` .
        ValueError: If `size` is not positive.
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.vision import Inter
        >>> decode_op = c_vision.Decode()
        >>> bbox_op = c_vision.ResizeWithBBox(50, Inter.NEAREST)
        >>> transforms_list = [decode_op, bbox_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    @check_resize_interpolation
    def __init__(self, size, interpolation=Inter.LINEAR):
        self.size = size
        self.interpolation = interpolation

    def parse(self):
        size = self.size
        if isinstance(size, int):
            size = (size,)
        return cde.ResizeWithBBoxOperation(size, DE_C_INTER_MODE.get(self.interpolation))


class RgbToBgr(ImageTensorOperation):
    """
    Convert RGB image to BGR.

    Raises:
        RuntimeError: If given tensor shape is not <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.vision import Inter
        >>>
        >>> decode_op = c_vision.Decode()
        >>> rgb2bgr_op = c_vision.RgbToBgr()
        >>> transforms_list = [decode_op, rgb2bgr_op]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision("ConvertColor")
    def __init__(self):
        super().__init__()

    def parse(self):
        return cde.RgbToBgrOperation()


class Rotate(ImageTensorOperation):
    """
    Rotate the input image by specified degrees.

    Args:
        degrees (Union[int, float]): Rotation degrees.

        resample (Inter, optional): An optional resampling filter. Default: ``Inter.NEAREST``.
            It can be ``Inter.BILINEAR``, ``Inter.NEAREST``, ``Inter.BICUBIC``.

            - ``Inter.BILINEA`` , means resample method is bilinear interpolation.
            - ``Inter.NEAREST`` , means resample method is nearest-neighbor interpolation.
            - ``Inter.BICUBIC`` , means resample method is bicubic interpolation.

        expand (bool, optional):  Optional expansion flag. Default: ``False``. If set to ``True``,
            expand the output image to make it large enough to hold the entire rotated image.
            If set to ``False`` or omitted, make the output image the same size as the input.
            Note that the expand flag assumes rotation around the center and no translation.
        center (tuple, optional): Optional center of rotation (a 2-tuple). Default: ``None``.
            Origin is the top left corner. ``None`` sets to the center of the image.
        fill_value (Union[int, tuple[int]], optional): Optional fill color for the area outside the rotated image.
            If it is a 3-tuple, it is used to fill R, G, B channels respectively.
            If it is an integer, it is used for all RGB channels.
            The fill_value values must be in range [0, 255]. Default: ``0``.

    Raises:
        TypeError: If `degrees` is not of type int or float.
        TypeError: If `resample` is not of type :class:`mindspore.dataset.vision.Inter` .
        TypeError: If `expand` is not of type bool.
        TypeError: If `center` is not of type tuple.
        TypeError: If `fill_value` is not of type int or tuple[int].
        ValueError: If `fill_value` is not in range [0, 255].
        RuntimeError: If given tensor shape is not <H, W> or <..., H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.vision import Inter
        >>> transforms_list = [c_vision.Decode(),
        ...                    c_vision.Rotate(degrees=30.0,
        ...                    resample=Inter.NEAREST,
        ...                    expand=True)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    @check_rotate
    def __init__(self, degrees, resample=Inter.NEAREST, expand=False, center=None, fill_value=0):
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

    def parse(self):
        return cde.RotateOperation(self.degrees, DE_C_INTER_MODE.get(self.resample), self.expand, self.center,
                                   self.fill_value)


class SlicePatches(ImageTensorOperation):
    """
    Slice Tensor to multiple patches in horizontal and vertical directions.

    The usage scenario is suitable to large height and width Tensor. The Tensor
    will keep the same if set both num_height and num_width to 1. And the
    number of output tensors is equal to num_height*num_width.

    Args:
        num_height (int, optional): The number of patches in vertical direction, which must be positive.
            Default: ``1``.
        num_width (int, optional): The number of patches in horizontal direction, which must be positive.
            Default: ``1``.
        slice_mode (Inter, optional): A mode represents pad or drop. Default: ``SliceMode.PAD``.
            It can be ``SliceMode.PAD``, ``SliceMode.DROP``.
        fill_value (int, optional): The border width in number of pixels in
            right and bottom direction if `slice_mode` is set to be ``SliceMode.PAD``.
            The fill_value must be in range [0, 255]. Default: ``0``.

    Raises:
        TypeError: If `num_height` is not of type int.
        TypeError: If `num_width` is not of type int.
        TypeError: If `slice_mode` is not of type :class:`mindspore.dataset.vision.Inter` .
        TypeError: If `fill_value` is not of type int.
        ValueError: If `num_height` is not positive.
        ValueError: If `num_width` is not positive.
        ValueError: If `fill_value` is not in range [0, 255].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> # default padding mode
        >>> decode_op = c_vision.Decode()
        >>> num_h, num_w = (1, 4)
        >>> slice_patches_op = c_vision.SlicePatches(num_h, num_w)
        >>> transforms_list = [decode_op, slice_patches_op]
        >>> cols = ['img' + str(x) for x in range(num_h*num_w)]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"],
        ...                                                 output_columns=cols)
    """

    @deprecated_c_vision()
    @check_slice_patches
    def __init__(self, num_height=1, num_width=1, slice_mode=SliceMode.PAD, fill_value=0):
        self.num_height = num_height
        self.num_width = num_width
        self.slice_mode = slice_mode
        self.fill_value = fill_value

    def parse(self):
        return cde.SlicePatchesOperation(self.num_height, self.num_width,
                                         DE_C_SLICE_MODE.get(self.slice_mode), self.fill_value)


class SoftDvppDecodeRandomCropResizeJpeg(ImageTensorOperation):
    """
    A combination of `Crop` , `Decode` and `Resize` using the simulation algorithm of Ascend series chip DVPP module.

    The usage scenario is consistent with SoftDvppDecodeResizeJpeg.
    The input image size should be in range [32*32, 8192*8192].
    The zoom-out and zoom-in multiples of the image length and width should in the range [1/32, 16].
    Only images with an even resolution can be output. The output of odd resolution is not supported.

    Note:
        SoftDvppDecodeRandomCropResizeJpeg is not supported as of 1.8 version.
        Please use RandomCropDecodeResize instead.

    Args:
        size (Union[int, Sequence[int]]): The size of the output image. The size value(s) must be positive.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, an image of size (height, width) will be cropped.
        scale (Union[list, tuple], optional): Range [min, max) of respective size of the
            original size to be cropped, which must be non-negative. Default: ``(0.08, 1.0)``.
        ratio (Union[list, tuple], optional): Range [min, max) of aspect ratio to be
            cropped, which must be non-negative. Default: ``(3. / 4., 4. / 3.)``.
        max_attempts (int, optional): The maximum number of attempts to propose a valid crop_area. Default: ``10``.
            If exceeded, fall back to use center_crop instead. The `max_attempts` value must be positive.

    Raises:
        TypeError: If `size` is not of type int or Sequence[int].
        TypeError: If `scale` is not of type tuple or list.
        TypeError: If `ratio` is not of type tuple or list.
        TypeError: If `max_attempts` is not of type int.
        ValueError: If `size` is not positive.
        ValueError: If `scale` is negative.
        ValueError: If `ratio` is negative.
        ValueError: If `max_attempts` is not positive.
        RuntimeError: If given tensor is not a 1D sequence.

    Supported Platforms:
        ``CPU``
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), max_attempts=10):
        raise NotImplementedError("SoftDvppDecodeRandomCropResizeJpeg is not supported as of 1.8 version. "
                                  "Please use RandomCropDecodeResize instead.")

    def parse(self):
        raise NotImplementedError("SoftDvppDecodeRandomCropResizeJpeg is not supported as of 1.8 version. "
                                  "Please use RandomCropDecodeResize instead.")


class SoftDvppDecodeResizeJpeg(ImageTensorOperation):
    """
    Decode and resize JPEG image using the simulation algorithm of Ascend series chip DVPP module.

    It is recommended to use this algorithm in the following scenarios:
    When training, the DVPP of the Ascend chip is not used,
    and the DVPP of the Ascend chip is used during inference,
    and the accuracy of inference is lower than the accuracy of training;
    and the input image size should be in range [32*32, 8192*8192].
    The zoom-out and zoom-in multiples of the image length and width should in the range [1/32, 16].
    Only images with an even resolution can be output. The output of odd resolution is not supported.

    Note:
        SoftDvppDecodeResizeJpeg is not supported as of 1.8 version. Please use Decode and Resize instead.

    Args:
        size (Union[int, Sequence[int]]): The output size of the resized image. The size value(s) must be positive.
            If size is an integer, smaller edge of the image will be resized to this value with
            the same image aspect ratio.
            If size is a sequence of length 2, an image of size (height, width) will be cropped.

    Raises:
        TypeError: If `size` is not of type int or Sequence[int].
        ValueError: If `size` is not positive.
        RuntimeError: If given tensor is not a 1D sequence.

    Supported Platforms:
        ``CPU``
    """

    def __init__(self, size):
        raise NotImplementedError("SoftDvppDecodeResizeJpeg is not supported as of 1.8 version. "
                                  "Please use Decode and Resize instead.")

    def parse(self):
        raise NotImplementedError("SoftDvppDecodeResizeJpeg is not supported as of 1.8 version. "
                                  "Please use Decode and Resize instead.")


class UniformAugment(ImageTensorOperation):
    """
    Perform randomly selected augmentation on input image.

    Args:
        transforms (TensorOperation): C++ transformation operation to be applied on random selection
            of bounding box regions of a given image (Python operations are not accepted).
        num_ops (int, optional): Number of operations to be selected and applied, which must be positive. Default: 2.

    Raises:
        TypeError: If `transform` is not an image processing operation
            in :class:`mindspore.dataset.vision.c_transforms` .
        TypeError: If `num_ops` is not of type int.
        ValueError: If `num_ops` is not positive.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import mindspore.dataset.vision.py_transforms as py_vision
        >>> transforms_list = [c_vision.RandomHorizontalFlip(),
        ...                    c_vision.RandomVerticalFlip(),
        ...                    c_vision.RandomColorAdjust(),
        ...                    c_vision.RandomRotation(degrees=45)]
        >>> uni_aug_op = c_vision.UniformAugment(transforms=transforms_list, num_ops=2)
        >>> transforms_all = [c_vision.Decode(), c_vision.Resize(size=[224, 224]),
        ...                   uni_aug_op]
        >>> image_folder_dataset_1 = image_folder_dataset.map(operations=transforms_all,
        ...                                                   input_columns="image",
        ...                                                   num_parallel_workers=1)
    """

    @deprecated_c_vision()
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


class VerticalFlip(ImageTensorOperation):
    """
    Flip the input image vertically.

    Raises:
        RuntimeError: If given tensor shape is not <H, W> or <..., H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> transforms_list = [c_vision.Decode(), c_vision.VerticalFlip()]
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image"])
    """

    @deprecated_c_vision()
    def __init__(self):
        super().__init__()

    def parse(self):
        return cde.VerticalFlipOperation()
