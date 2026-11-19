# Copyright 2019-2024 Huawei Technologies Co., Ltd
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
    >>> import mindspore.dataset as ds
    >>> import mindspore.dataset.vision as vision
    >>> from mindspore.dataset.vision import Border, Inter
    >>> import mindspore.dataset.transforms as transforms
    >>>
    >>> image_folder_dataset_dir = "/path/to/image_folder_dataset_directory"
    >>> # create a dataset that reads all files in dataset_dir with 8 threads
    >>> image_folder_dataset = ds.ImageFolderDataset(image_folder_dataset_dir,
    ...                                              num_parallel_workers=8)
    >>> # create a list of transformations to be applied to the image data
    >>> transforms_list = [vision.Decode(),
    ...                    vision.Resize((256, 256), interpolation=Inter.LINEAR),
    ...                    vision.RandomCrop(200, padding_mode=Border.EDGE),
    ...                    vision.Normalize((100, 115.0, 121.0), (71.0, 68.0, 70.0)),
    ...                    vision.HWC2CHW()]
    >>> onehot_op = transforms.transforms.OneHot(num_classes=10)
    >>> # apply the transformation to the dataset through data1.map()
    >>> image_folder_dataset = image_folder_dataset.map(operations=transforms_list,
    ...                                                 input_columns="image")
    >>> image_folder_dataset = image_folder_dataset.map(operations=onehot_op,
    ...                                                 input_columns="label")
"""

# pylint: disable=too-few-public-methods
import numbers
import random
import numpy as np
from PIL import Image

import mindspore._c_dataengine as cde
from mindspore._c_expression import typing
from . import py_transforms_util as util
from .py_transforms_util import is_pil
from .utils import AutoAugmentPolicy, Border, ImageBatchFormat, Inter, parse_padding
from .validators import (
    check_adjust_gamma,
    check_alpha,
    check_auto_augment,
    check_bounding_box_augment_cpp,
    check_center_crop,
    check_crop,
    check_cut_mix_batch_c,
    check_decode,
    check_erase,
    check_five_crop,
    check_linear_transform,
    check_mix_up,
    check_mix_up_batch_c,
    check_normalize,
    check_normalizepad,
    check_num_channels,
    check_pad,
    check_perspective,
    check_posterize,
    check_prob,
    check_rand_augment,
    check_random_color_adjust,
    check_random_crop,
    check_random_erasing,
    check_random_perspective,
    check_random_posterize,
    check_random_resize_crop,
    check_random_solarize,
    check_range,
    check_rescale,
    check_resize,
    check_resize_interpolation,
    check_solarize,
    check_ten_crop,
    check_trivial_augment_wide,
    check_to_tensor,
    check_device_target,
    FLOAT_MAX_INTEGER,
)
from ..core.datatypes import mstype_to_detype, nptype_to_detype
from ..transforms.py_transforms_util import Implementation
from ..transforms.transforms import PyTensorOperation, TensorOperation, TypeCast


class ImageTensorOperation(TensorOperation):
    """
    Base class of Image Tensor Ops.
    """

    def __call__(self, *input_tensor_list):
        for tensor in input_tensor_list:
            if not isinstance(tensor, (np.ndarray, Image.Image)):
                raise TypeError(
                    "Input should be NumPy or PIL image, got {}.".format(type(tensor)))
        return super().__call__(*input_tensor_list)

    def parse(self):
        # Note: subclasses must implement `def parse(self)` so do not make ImageTensorOperation's parse a staticmethod.
        raise NotImplementedError("ImageTensorOperation has to implement parse() method.")


class VideoTensorOperation(TensorOperation):
    """
    Base class of Video Tensor Ops
    """

    def __call__(self, *input_tensor_list):
        for tensor in input_tensor_list:
            if not isinstance(tensor, np.ndarray):
                raise TypeError(
                    "Input should be ndarray, got {}.".format(type(tensor)))
        return super().__call__(*input_tensor_list)

    def parse(self):
        raise NotImplementedError("VideoTensorOperation has to implement parse() method.")


class AdjustGamma(ImageTensorOperation, PyTensorOperation):
    r"""
    Apply gamma correction on input image. Input image is expected to be in <..., H, W, C> or <H, W> format.

    .. math::
        I_{\text{out}} = 255 \times \text{gain} \times \left(\frac{I_{\text{in}}}{255}\right)^{\gamma}

    See `Gamma Correction`_ for more details.

    .. _Gamma Correction: https://en.wikipedia.org/wiki/Gamma_correction

    Args:
        gamma (float): Non-negative real number.
            The output image pixel value is exponentially related to the input image pixel value.
            `gamma` larger than ``1.0`` makes the shadows darker,
            while `gamma` smaller than ``1.0`` makes dark regions lighter.
        gain (float, optional): The constant multiplier. Default: ``1.0``.

    Raises:
        TypeError: If `gain` is not of type float.
        TypeError: If `gamma` is not of type float.
        ValueError: If `gamma` is less than ``0.0``.
        RuntimeError: If given tensor shape is not <H, W> or <..., H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> transforms_list = [vision.AdjustGamma(gamma=10.0, gain=1.0)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (100, 100, 3) uint8
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dtype=np.uint8).reshape((2, 2, 3))
        >>> output = vision.AdjustGamma(gamma=0.1, gain=1.0)(data)
        >>> print(output.shape, output.dtype)
        (2, 2, 3) uint8

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """

    @check_adjust_gamma
    def __init__(self, gamma, gain=1):
        super().__init__()
        self.gamma = gamma
        self.gain = gain
        self.random = False

    def parse(self):
        return cde.AdjustGammaOperation(self.gamma, self.gain)

    def _execute_py(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be gamma adjusted.

        Returns:
            PIL Image, gamma adjusted image.
        """
        return util.adjust_gamma(img, self.gamma, self.gain)


class AutoAugment(ImageTensorOperation):
    """
    Apply AutoAugment data augmentation method based on
    `AutoAugment: Learning Augmentation Strategies from Data <https://arxiv.org/pdf/1805.09501.pdf>`_ .
    This operation works only with 3-channel RGB images.

    Args:
        policy (AutoAugmentPolicy, optional): AutoAugment policies learned on different datasets.
            Default: ``AutoAugmentPolicy.IMAGENET``.
            It can be ``AutoAugmentPolicy.IMAGENET``, ``AutoAugmentPolicy.CIFAR10``, ``AutoAugmentPolicy.SVHN``.
            Randomly apply 2 operations from a candidate set. See auto augmentation details in AutoAugmentPolicy.

            - ``AutoAugmentPolicy.IMAGENET``, means to apply AutoAugment learned on ImageNet dataset.

            - ``AutoAugmentPolicy.CIFAR10``, means to apply AutoAugment learned on Cifar10 dataset.

            - ``AutoAugmentPolicy.SVHN``, means to apply AutoAugment learned on SVHN dataset.

        interpolation (Inter, optional): Image interpolation method defined by :class:`~.vision.Inter` .
            Default: ``Inter.NEAREST``.
        fill_value (Union[int, tuple[int]], optional): Pixel fill value for the area outside the transformed image.
            It can be an int or a 3-tuple. Default: ``0``.

            - If it is a 3-tuple, it is used to fill R, G, B channels respectively.
            - If it is an integer, it is used for all RGB channels. The fill_value values must be in the range [0, 255].

    Raises:
        TypeError: If `policy` is not of type :class:`mindspore.dataset.vision.AutoAugmentPolicy` .
        TypeError: If `interpolation` is not of type :class:`~.vision.Inter` .
        TypeError: If `fill_value` is not an integer or a tuple of length 3.
        RuntimeError: If given tensor shape is not <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>> from mindspore.dataset.vision import AutoAugmentPolicy, Inter
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> transforms_list = [vision.AutoAugment(policy=AutoAugmentPolicy.IMAGENET,
        ...                                       interpolation=Inter.NEAREST,
        ...                                       fill_value=0)]
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (100, 100, 3) uint8
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
        >>> output = vision.AutoAugment()(data)
        >>> print(output.shape, output.dtype)
        (100, 100, 3) uint8

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
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


class BoundingBoxAugment(ImageTensorOperation):
    """
    Apply a given image processing operation on a random selection of bounding box regions of a given image.

    Args:
        transform (TensorOperation): Transformation operation to be applied on random selection
            of bounding box regions of a given image.
        ratio (float, optional): Ratio of bounding boxes to apply augmentation on.
            Range: [0.0, 1.0]. Default: ``0.3``.

    Raises:
        TypeError: If `transform` is not an image processing operation in `mindspore.dataset.vision`.
        TypeError: If `ratio` is not of type float.
        ValueError: If `ratio` is not in the range [0.0, 1.0].
        RuntimeError: If given bounding box is invalid.

    Supported Platforms:
        ``CPU``

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


class CenterCrop(ImageTensorOperation, PyTensorOperation):
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
        RuntimeError: If given tensor shape is not <H, W> or <..., H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>>
        >>> # crop image to a square
        >>> transforms_list1 = [vision.CenterCrop(50)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list1, input_columns=["image"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (50, 50, 3) uint8
        >>>
        >>> # crop image to portrait style
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> transforms_list2 = [vision.CenterCrop((60, 40))]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list2, input_columns=["image"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (60, 40, 3) uint8
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dtype=np.uint8).reshape((2, 2, 3))
        >>> output = vision.CenterCrop(1)(data)
        >>> print(output.shape, output.dtype)
        (1, 1, 3) uint8

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
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

    def _execute_py(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be center cropped.

        Returns:
            PIL Image, cropped image.
        """
        return util.center_crop(img, self.size)


class Crop(ImageTensorOperation):
    """
    Crop the input image at a specific location.

    Supports Ascend hardware acceleration and can be enabled through the `.device("Ascend")` method.

    Args:
        coordinates (sequence): Coordinates of the upper left corner of the cropping image. Must be a sequence of two
            values, in the form of (top, left).
        size (Union[int, sequence]): The output size of the cropped image.

            - If size is an integer, a square crop of size (size, size) is returned.
            - If size is a sequence of length 2, it should be (height, width).

            The size value(s) must be larger than 0.

    Raises:
        TypeError: If `coordinates` is not of type sequence.
        TypeError: If `size` is not of type integer or sequence.
        ValueError: If `coordinates` is less than 0.
        ValueError: If `size` is less than or equal to 0.
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU`` ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> crop_op = vision.Crop((0, 0), 32)
        >>> transforms_list = [crop_op]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (32, 32, 3) uint8
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dtype=np.uint8).reshape((2, 2, 3))
        >>> output = vision.Crop((0, 0), 1)(data)
        >>> print(output.shape, output.dtype)
        (1, 1, 3) uint8

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """

    @check_crop
    def __init__(self, coordinates, size):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.coordinates = coordinates
        self.size = size
        self.implementation = Implementation.C

    @check_device_target
    def device(self, device_target="CPU"):
        """
        Set the device for the current operator execution.

        When the device is Ascend, input/output shape should be limited from [4, 6] to [32768, 32768].

        Args:
            device_target (str, optional): The operator will be executed on this device. Currently supports
                ``"CPU"`` and ``"Ascend"`` . Default: ``"CPU"`` .

        Raises:
            TypeError: If `device_target` is not of type str.
            ValueError: If `device_target` is not within the valid set of ["CPU", "Ascend"].

        Supported Platforms:
            ``CPU`` ``Ascend``

        Examples:
            >>> import numpy as np
            >>> import mindspore.dataset as ds
            >>> import mindspore.dataset.vision as vision
            >>>
            >>> # Use the transform in dataset pipeline mode
            >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
            >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
            >>> crop_op = vision.Crop((0, 0), (100, 75)).device("Ascend")
            >>> transforms_list = [crop_op]
            >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
            >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            ...     print(item["image"].shape, item["image"].dtype)
            ...     break
            (100, 75, 3) uint8
            >>>
            >>> # Use the transform in eager mode
            >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
            >>> output = vision.Crop((0, 0), 64).device("Ascend")(data)
            >>> print(output.shape, output.dtype)
            (64, 64, 3) uint8

        Tutorial Examples:
            - `Illustration of vision transforms
              <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
        """
        self.device_target = device_target
        return self

    def parse(self):
        return cde.CropOperation(self.coordinates, self.size, self.device_target)


class CutMixBatch(ImageTensorOperation):
    """
    Apply CutMix transformation on input batch of images and labels.
    Note that you need to make labels into one-hot format and batched before calling this operation.

    Args:
        image_batch_format (ImageBatchFormat): The format of the image batch. Can be any of
            [ImageBatchFormat.NHWC, ImageBatchFormat.NCHW].
        alpha (float, optional): Hyperparameter of beta distribution, must be larger than 0. Default: ``1.0``.
        prob (float, optional): The probability by which CutMix is applied to each image,
            which must be in the range [0.0, 1.0]. Default: ``1.0``.

    Raises:
        TypeError: If `image_batch_format` is not of type :class:`mindspore.dataset.vision.ImageBatchFormat` .
        TypeError: If `alpha` is not of type float.
        TypeError: If `prob` is not of type float.
        ValueError: If `alpha` is less than or equal to 0.
        ValueError: If `prob` is not in the range [0.0, 1.0].
        RuntimeError: If given tensor shape is not <H, W, C>.

    Supported Platforms:
        ``CPU``

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


class Decode(ImageTensorOperation, PyTensorOperation):
    """
    Decode the input image in RGB mode.
    Supported image formats: JPEG, BMP, PNG, TIFF, GIF (need `to_pil=True`), WEBP (need `to_pil=True`).

    Supports Ascend hardware acceleration and can be enabled through the `.device("Ascend")` method.

    Args:
        to_pil (bool, optional): Whether to decode the image to the PIL data type. If ``True``,
            the image will be decoded to the PIL data type, otherwise it will be decoded to the
            NumPy data type. Default: ``False``.

    Raises:
        RuntimeError: If given tensor is not a 1D sequence.
        RuntimeError: If the input is not raw image bytes.
        RuntimeError: If the input image is already decoded.

    Supported Platforms:
        ``CPU`` ``Ascend``

    Examples:
        >>> import os
        >>> import numpy as np
        >>> from PIL import Image, ImageDraw
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> class MyDataset:
        ...     def __init__(self):
        ...         self.data = []
        ...         img = Image.new("RGB", (300, 300), (255, 255, 255))
        ...         draw = ImageDraw.Draw(img)
        ...         draw.ellipse(((0, 0), (100, 100)), fill=(255, 0, 0), outline=(255, 0, 0), width=5)
        ...         img.save("./1.jpg")
        ...         data = np.fromfile("./1.jpg", np.uint8)
        ...         self.data.append(data)
        ...
        ...     def __getitem__(self, index):
        ...         return self.data[0]
        ...
        ...     def __len__(self):
        ...         return 5
        >>>
        >>> my_dataset = MyDataset()
        >>> generator_dataset = ds.GeneratorDataset(my_dataset, column_names="image")
        >>> transforms_list = [vision.Decode(), vision.RandomHorizontalFlip()]
        >>> generator_dataset = generator_dataset.map(operations=transforms_list, input_columns=["image"])
        >>> for item in generator_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (300, 300, 3) uint8
        >>> os.remove("./1.jpg")
        >>>
        >>> # Use the transform in eager mode
        >>> img = Image.new("RGB", (300, 300), (255, 255, 255))
        >>> draw = ImageDraw.Draw(img)
        >>> draw.polygon([(50, 50), (150, 50), (100, 150)], fill=(0, 255, 0), outline=(0, 255, 0))
        >>> img.save("./2.jpg")
        >>> data = np.fromfile("./2.jpg", np.uint8)
        >>> output = vision.Decode()(data)
        >>> print(output.shape, output.dtype)
        (300, 300, 3) uint8
        >>> os.remove("./2.jpg")

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """

    @check_decode
    def __init__(self, to_pil=False):
        super().__init__()
        self.to_pil = to_pil
        if to_pil:
            self.implementation = Implementation.PY
        else:
            self.implementation = Implementation.C

    def __call__(self, img):
        """
        Call method for input conversion for eager mode with C++ implementation.
        """
        if isinstance(img, bytes):
            img = np.frombuffer(img, dtype=np.uint8)
        if not isinstance(img, np.ndarray):
            raise TypeError("The type of the encoded image should be {0}, but got {1}.".format(np.ndarray, type(img)))
        if img.dtype.type is np.str_:
            raise TypeError("The data type of the encoded image can not be {}.".format(img.dtype.type))
        if img.ndim != 1:
            raise TypeError("The number of array dimensions of the encoded image should be 1, "
                            "but got {0}.".format(img.ndim))
        return super().__call__(img)

    @check_device_target
    def device(self, device_target="CPU"):
        """
        Set the device for the current operator execution.

        When the device is Ascend, input type only supports `uint8`.

        Args:
            device_target (str, optional): The operator will be executed on this device. Currently supports
                ``"CPU"`` and ``"Ascend"`` . Default: ``"CPU"`` .

        Raises:
            TypeError: If `device_target` is not of type str.
            ValueError: If `device_target` is not within the valid set of ["CPU", "Ascend"].

        Supported Platforms:
            ``CPU`` ``Ascend``

        Examples:
            >>> import os
            >>> import numpy as np
            >>> from PIL import Image, ImageDraw
            >>> import mindspore.dataset as ds
            >>> import mindspore.dataset.vision as vision
            >>> from mindspore.dataset.vision import Inter
            >>>
            >>> # Use the transform in dataset pipeline mode
            >>> class MyDataset:
            ...     def __init__(self):
            ...         self.data = []
            ...         img = Image.new("RGB", (300, 300), (255, 255, 255))
            ...         draw = ImageDraw.Draw(img)
            ...         draw.ellipse(((0, 0), (100, 100)), fill=(255, 0, 0), outline=(255, 0, 0), width=5)
            ...         img.save("./1.jpg")
            ...         data = np.fromfile("./1.jpg", np.uint8)
            ...         self.data.append(data)
            ...
            ...     def __getitem__(self, index):
            ...         return self.data[0]
            ...
            ...     def __len__(self):
            ...         return 5
            >>>
            >>> my_dataset = MyDataset()
            >>> generator_dataset = ds.GeneratorDataset(my_dataset, column_names="image")
            >>> decode_op = vision.Decode().device("Ascend")
            >>> resize_op = vision.Resize([100, 75], Inter.BICUBIC)
            >>> transforms_list = [decode_op, resize_op]
            >>> generator_dataset = generator_dataset.map(operations=transforms_list, input_columns=["image"])
            >>> for item in generator_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            ...     print(item["image"].shape, item["image"].dtype)
            ...     break
            (100, 75, 3) uint8
            >>> os.remove("./1.jpg")
            >>>
            >>> # Use the transform in eager mode
            >>> img = Image.new("RGB", (300, 300), (255, 255, 255))
            >>> draw = ImageDraw.Draw(img)
            >>> draw.polygon([(50, 50), (150, 50), (100, 150)], fill=(0, 255, 0), outline=(0, 255, 0))
            >>> img.save("./2.jpg")
            >>> data = np.fromfile("./2.jpg", np.uint8)
            >>> output = vision.Decode().device("Ascend")(data)
            >>> print(output.shape, output.dtype)
            (300, 300, 3) uint8
            >>> os.remove("./2.jpg")

        Tutorial Examples:
            - `Illustration of vision transforms
              <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
        """
        if self.implementation == Implementation.PY and device_target == "Ascend":
            raise ValueError("The transform \"Decode(to_pil=True)\" cannot be performed on Ascend device, " +
                             "please set \"to_pil=False\".")

        self.device_target = device_target
        return self

    def parse(self):
        return cde.DecodeOperation(True, self.device_target)

    def _execute_py(self, img):
        """
        Execute method.

        Args:
            img (NumPy): Image to be decoded.

        Returns:
            img (NumPy, PIL Image), Decoded image.
        """
        return util.decode(img)


class DecodeVideo(VideoTensorOperation):
    """
    Decode the input raw video and audio bytes.

    Supported video formats: AVI, H264, H265, MOV, MP4, WMV.

    Raises:
        RuntimeError: If the input ndarray is not a 1D array.
        RuntimeError: If the data type of the elements is not uint8.
        RuntimeError: If the input ndarray is empty.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> # Custom class to generate and read video dataset
        >>> class VideoDataset:
        ...     def __init__(self, file_list):
        ...         self.file_list = file_list
        ...
        ...     def __getitem__(self, index):
        ...         filename = self.file_list[index]
        ...         return np.fromfile(filename, np.uint8)
        ...
        ...     def __len__(self):
        ...         return len(self.file_list)
        >>>
        >>> dataset = ds.GeneratorDataset(VideoDataset(["/path/to/video/file"]), ["data"])
        >>> decode_video = vision.DecodeVideo()
        >>> dataset = dataset.map(operations=[decode_video], input_columns=["data"], output_columns=["video", "audio"])
        >>>
        >>> # Use the transform in eager mode
        >>> filename = "/path/to/video/file"
        >>> raw_ndarray = np.fromfile(filename, np.uint8)
        >>> mindspore_output = vision.DecodeVideo()(raw_ndarray)
    """

    def __init__(self):
        super().__init__()
        self.implementation = Implementation.C

    def parse(self):
        return cde.DecodeVideoOperation()


class Erase(ImageTensorOperation):
    """
    Erases a specified area of the input image and fills it with the specified pixel value `value`.

    Supports Ascend hardware acceleration and can be enabled through the `.device("Ascend")` method.

    Args:
        top (int): Vertical ordinate of the upper left corner of erased region.
        left (int): Horizontal ordinate of the upper left corner of erased region.
        height (int): Height of erased region.
        width (int): Width of erased region.
        value (Union[float, Sequence[float, float, float]], optional): Pixel value used to pad the erased area.
            Default: ``0.0``.

            - If float is provided, it will be used for all RGB channels.
            - If Sequence[float, float, float] is provided, it will be used for R, G, B channels respectively.

        inplace (bool, optional): Whether to apply erasing inplace. Default: ``False``.

    Raises:
        TypeError: If `top` is not of type int.
        ValueError: If `top` is negative.
        TypeError: If `left` is not of type int.
        ValueError: If `left` is negative.
        TypeError: If `height` is not of type int.
        ValueError: If `height` is not positive.
        TypeError: If `width` is not of type int.
        ValueError: If `width` is not positive.
        TypeError: If `value` is not of type float or Sequence[float, float, float].
        ValueError: If `value` is not in the range [0, 255].
        TypeError: If `inplace` is not of type bool.
        RuntimeError: If shape of the input image is not <H, W, C>.

    Supported Platforms:
        ``CPU`` ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> transforms_list = [vision.Erase(10,10,10,10)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (100, 100, 3) uint8
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
        >>> output = vision.Erase(10, 10, 10, 10)(data)
        >>> print(output.shape, output.dtype)
        (100, 100, 3) uint8

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """

    @check_erase
    def __init__(self, top, left, height, width, value=0.0, inplace=False):
        super().__init__()
        self.top = top
        self.left = left
        self.height = height
        self.width = width
        if isinstance(value, (int, float)):
            value = tuple([value])
        self.value = value
        self.inplace = inplace

    @check_device_target
    def device(self, device_target="CPU"):
        """
        Set the device for the current operator execution.

        When the device is Ascend, input type supports `uint8` or `float32` , input channel supports 1 and 3.
        The input data has a height limit of [4, 8192] and a width limit of [6, 4096].
        The inplace parameter is not supported.

        Args:
            device_target (str, optional): The operator will be executed on this device. Currently supports
                ``"CPU"`` and ``"Ascend"`` . Default: ``"CPU"`` .

        Raises:
            TypeError: If `device_target` is not of type str.
            ValueError: If `device_target` is not within the valid set of ["CPU", "Ascend"].

        Supported Platforms:
            ``CPU`` ``Ascend``

        Examples:
            >>> import numpy as np
            >>> import mindspore.dataset as ds
            >>> import mindspore.dataset.vision as vision
            >>>
            >>> # Use the transform in dataset pipeline mode
            >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
            >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
            >>> transforms_list = [vision.Erase(10, 10, 10, 10, (100, 100, 100)).device("Ascend")]
            >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
            >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            ...     print(item["image"].shape, item["image"].dtype)
            ...     break
            (100, 100, 3) uint8
            >>>
            >>> # Use the transform in eager mode
            >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
            >>> output = vision.Erase(10, 10, 10, 10, (100, 100, 100)).device("Ascend")(data)
            >>> print(output.shape, output.dtype)
            (100, 100, 3) uint8

        Tutorial Examples:
            - `Illustration of vision transforms
              <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
        """
        self.device_target = device_target
        return self

    def parse(self):
        return cde.EraseOperation(self.top, self.left, self.height, self.width, self.value, self.inplace,
                                  self.device_target)


class FiveCrop(PyTensorOperation):
    """
    Crops the given image into one central crop and four corners.

    Args:
        size (Union[int, Sequence[int, int]]): The size of the cropped image.
            If a single integer is provided, a square of size (size, size) will be cropped with this value.
            If a Sequence of length 2 is provided, an image of size (height, width) will be cropped.

    Raises:
        TypeError: If `size` is not of type integer or Sequence of integer.
        ValueError: If `size` is not positive.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import os
        >>> import numpy as np
        >>> from PIL import Image, ImageDraw
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>> from mindspore.dataset.transforms import Compose
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> class MyDataset:
        ...     def __init__(self):
        ...         self.data = []
        ...         img = Image.new("RGB", (300, 300), (255, 255, 255))
        ...         draw = ImageDraw.Draw(img)
        ...         draw.ellipse(((0, 0), (100, 100)), fill=(255, 0, 0), outline=(255, 0, 0), width=5)
        ...         img.save("./1.jpg")
        ...         data = np.fromfile("./1.jpg", np.uint8)
        ...         self.data.append(data)
        ...
        ...     def __getitem__(self, index):
        ...         return self.data[0]
        ...
        ...     def __len__(self):
        ...         return 5
        >>>
        >>> my_dataset = MyDataset()
        >>> generator_dataset = ds.GeneratorDataset(my_dataset, column_names="image")
        >>> transforms_list = Compose([vision.Decode(to_pil=True),
        ...                            vision.FiveCrop(size=200),
        ...                            # 4D stack of 5 images
        ...                            lambda *images: np.stack([vision.ToTensor()(image) for image in images])])
        >>> # apply the transform to dataset through map function
        >>> generator_dataset = generator_dataset.map(operations=transforms_list, input_columns="image")
        >>> for item in generator_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (5, 3, 200, 200) float32
        >>> os.remove("./1.jpg")
        >>>
        >>> # Use the transform in eager mode
        >>> img = Image.new("RGB", (300, 300), (255, 255, 255))
        >>> draw = ImageDraw.Draw(img)
        >>> draw.polygon([(50, 50), (150, 50), (100, 150)], fill=(0, 255, 0), outline=(0, 255, 0))
        >>> img.save("./2.jpg")
        >>> data = Image.open("./2.jpg")
        >>> output = vision.FiveCrop(size=20)(data)
        >>> for cropped_img in output:
        ...     print(cropped_img.size)
        ...     break
        (20, 20)
        >>> os.remove("./2.jpg")


    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """

    @check_five_crop
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.random = False
        self.implementation = Implementation.PY

    def _execute_py(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            tuple, a tuple of five PIL Image in order of top_left, top_right, bottom_left, bottom_right, center.
        """
        return util.five_crop(img, self.size)


class Grayscale(PyTensorOperation):
    """
    Convert the input PIL Image to grayscale.

    Args:
        num_output_channels (int): The number of channels desired for the output image, must be ``1`` or ``3``.
            If ``3`` is provided, the returned image will have 3 identical RGB channels. Default: ``1``.

    Raises:
        TypeError: If `num_output_channels` is not of type integer.
        ValueError: If `num_output_channels` is not ``1`` or ``3``.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import os
        >>> import numpy as np
        >>> from PIL import Image, ImageDraw
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>> from mindspore.dataset.transforms import Compose
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> class MyDataset:
        ...     def __init__(self):
        ...         self.data = []
        ...         img = Image.new("RGB", (300, 300), (255, 255, 255))
        ...         draw = ImageDraw.Draw(img)
        ...         draw.ellipse(((0, 0), (100, 100)), fill=(255, 0, 0), outline=(255, 0, 0), width=5)
        ...         img.save("./1.jpg")
        ...         data = np.fromfile("./1.jpg", np.uint8)
        ...         self.data.append(data)
        ...
        ...     def __getitem__(self, index):
        ...         return self.data[0]
        ...
        ...     def __len__(self):
        ...         return 5
        >>>
        >>> my_dataset = MyDataset()
        >>> generator_dataset = ds.GeneratorDataset(my_dataset, column_names="image")
        >>> transforms_list = Compose([vision.Decode(to_pil=True),
        ...                            vision.Grayscale(3),
        ...                            vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> generator_dataset = generator_dataset.map(operations=transforms_list, input_columns="image")
        >>> for item in generator_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (3, 300, 300) float32
        >>> os.remove("./1.jpg")
        >>>
        >>> # Use the transform in eager mode
        >>> img = Image.new("RGB", (300, 300), (255, 255, 255))
        >>> draw = ImageDraw.Draw(img)
        >>> draw.polygon([(50, 50), (150, 50), (100, 150)], fill=(0, 255, 0), outline=(0, 255, 0))
        >>> img.save("./2.jpg")
        >>> data = Image.open("./2.jpg")
        >>> output = vision.Grayscale(3)(data)
        >>> print(np.array(output).shape, np.array(output).dtype)
        (300, 300, 3) uint8
        >>> os.remove("./2.jpg")

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """

    @check_num_channels
    def __init__(self, num_output_channels=1):
        super().__init__()
        self.num_output_channels = num_output_channels
        self.random = False
        self.implementation = Implementation.PY

    def _execute_py(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image, converted grayscale image.
        """
        return util.grayscale(img, num_output_channels=self.num_output_channels)


class HWC2CHW(ImageTensorOperation):
    """
    Transposes the input image from shape <H, W, C> to <C, H, W>.
    If the input image is of shape <H, W>, it will remain unchanged.

    Note:
        This operation is executed on the CPU by default, but it can also be executed
        on the GPU or Ascend via heterogeneous acceleration.

    Raises:
        RuntimeError: If shape of the input image is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU`` ``GPU`` ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> transforms_list = [vision.RandomHorizontalFlip(0.75),
        ...                    vision.RandomCrop(64),
        ...                    vision.HWC2CHW()]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (3, 64, 64) uint8
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dtype=np.uint8).reshape((2, 2, 3))
        >>> output = vision.HWC2CHW()(data)
        >>> print(output.shape, output.dtype)
        (3, 2, 2) uint8

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """

    def __init__(self):
        super().__init__()
        self.implementation = Implementation.C
        self.random = False

    def parse(self):
        return cde.HwcToChwOperation()


class LinearTransformation(PyTensorOperation):
    r"""
    Linearly transform the input numpy.ndarray image with a square transformation matrix and a mean vector.

    It will first flatten the input image and subtract the mean vector from it, then compute the dot
    product with the transformation matrix, finally reshape it back to its original shape.

    Args:
        transformation_matrix (numpy.ndarray): A square transformation matrix in shape of (D, D), where
            :math:`D = C \times H \times W` .
        mean_vector (numpy.ndarray): A mean vector in shape of (D,), where :math:`D = C \times H \times W` .

    Raises:
        TypeError: If `transformation_matrix` is not of type `numpy.ndarray` .
        TypeError: If `mean_vector` is not of type `numpy.ndarray` .

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>> from mindspore.dataset.transforms import Compose
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> height, width = 32, 32
        >>> dim = 3 * height * width
        >>> transformation_matrix = np.ones([dim, dim])
        >>> mean_vector = np.zeros(dim)
        >>> transforms_list = Compose([vision.Resize((height,width)),
        ...                            vision.ToTensor(),
        ...                            vision.LinearTransformation(transformation_matrix, mean_vector)])
        >>> # apply the transform to dataset through map function
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns="image")
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (3, 32, 32) float64
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.random.randn(10, 10, 3)
        >>> transformation_matrix = np.random.randn(300, 300)
        >>> mean_vector = np.random.randn(300,)
        >>> output = vision.LinearTransformation(transformation_matrix, mean_vector)(data)
        >>> print(output.shape, output.dtype)
        (10, 10, 3) float64

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """

    @check_linear_transform
    def __init__(self, transformation_matrix, mean_vector):
        super().__init__()
        self.transformation_matrix = transformation_matrix
        self.mean_vector = mean_vector
        self.random = False
        self.implementation = Implementation.PY

    def _execute_py(self, np_img):
        """
        Execute method.

        Args:
            np_img (numpy.ndarray): Image in shape of <C, H, W> to be linearly transformed.

        Returns:
            numpy.ndarray, linearly transformed image.
        """
        return util.linear_transform(np_img, self.transformation_matrix, self.mean_vector)


class MixUp(PyTensorOperation):
    """
    Randomly mix up a batch of numpy.ndarray images together with its labels.

    Each image will be multiplied by a random weight :math:`lambda` generated from the Beta distribution and then added
    to another image multiplied by :math:`1 - lambda`. The same transformation will be applied to their labels with the
    same value of :math:`lambda`. Make sure that the labels are one-hot encoded in advance.

    Args:
        batch_size (int): The number of images in a batch.
        alpha (float): The alpha and beta parameters for the Beta distribution.
        is_single (bool, optional): If ``True``, it will randomly mix up [img0, ..., img(n-1), img(n)] with
            [img1, ..., img(n), img0] in each batch. Otherwise, it will randomly mix up images with the
            output of the previous batch. Default: ``True``.

    Raises:
        TypeError: If `batch_size` is not of type integer.
        TypeError: If `alpha` is not of type float.
        TypeError: If `is_single` is not of type boolean.
        ValueError: If `batch_size` is not positive.
        ValueError: If `alpha` is not positive.

    Supported Platforms:
        ``CPU``

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
        Call method to apply mix up transformation to image and label.

        Note: No execute method for MixUp

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


class MixUpBatch(ImageTensorOperation):
    """
    Apply MixUp transformation on input batch of images and labels. Each image is
    multiplied by a random weight (lambda) and then added to a randomly selected image from the batch
    multiplied by (1 - lambda). The same formula is also applied to the one-hot labels.

    The lambda is generated based on the specified alpha value. Two coefficients x1, x2 are randomly generated
    in the range [alpha, 1], and lambda = (x1 / (x1 + x2)).

    Note that you need to make labels into one-hot format and batched before calling this operation.

    Args:
        alpha (float, optional): Hyperparameter of Beta distribution. The value must be positive.
            Default: ``1.0``.

    Raises:
        TypeError: If `alpha` is not of type float.
        ValueError: If `alpha` is not positive.
        RuntimeError: If given tensor shape is not <N, H, W, C> or <N, C, H, W>.

    Supported Platforms:
        ``CPU``

    """

    @check_mix_up_batch_c
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.implementation = Implementation.C

    def parse(self):
        return cde.MixUpBatchOperation(self.alpha)


class Normalize(ImageTensorOperation):
    """
    Normalize the input image with respect to mean and standard deviation. This operation will normalize
    the input image with: output[channel] = (input[channel] - mean[channel]) / std[channel], where channel >= 1.

    Supports Ascend hardware acceleration and can be enabled through the `.device("Ascend")` method.

    Note:
        This operation is executed on the CPU by default, but it can also be executed
        on the GPU or Ascend via heterogeneous acceleration.

    Args:
        mean (sequence): List or tuple of mean values for each channel, with respect to channel order.
            The mean values must be in range [0.0, 255.0].
        std (sequence): List or tuple of standard deviations for each channel, with respect to channel order.
            The standard deviation values must be in range (0.0, 255.0].
        is_hwc (bool, optional): Whether the input image is in HWC format.
            ``True`` - HWC format, ``False`` - CHW format. Default: ``True``.

    Raises:
        TypeError: If `mean` is not of type sequence.
        TypeError: If `std` is not of type sequence.
        TypeError: If `is_hwc` is not of type bool.
        ValueError: If `mean` is not in the range [0.0, 255.0].
        ValueError: If `std` is not in the range (0.0, 255.0].
        RuntimeError: If given tensor shape is not <H, W> or <..., H, W, C>.

    Supported Platforms:
        ``CPU`` ``GPU`` ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> normalize_op = vision.Normalize(mean=[121.0, 115.0, 100.0], std=[70.0, 68.0, 71.0], is_hwc=True)
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=[normalize_op],
        ...                                                 input_columns=["image"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (100, 100, 3) float32
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
        >>> output = vision.Normalize(mean=[121.0, 115.0, 100.0], std=[70.0, 68.0, 71.0])(data)
        >>> print(output.shape, output.dtype)
        (100, 100, 3) float32

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """

    @check_normalize
    def __init__(self, mean, std, is_hwc=True):
        super().__init__()
        self.mean = mean
        self.std = std
        self.is_hwc = is_hwc
        self.random = False
        self.implementation = Implementation.C

    @check_device_target
    def device(self, device_target="CPU"):
        """
        Set the device for the current operator execution.

        - When the device is CPU, input type supports `uint8`/`float32`/`float64`, input channel supports 1/2/3.
        - When the device is Ascend, input type supports `uint8`/`float32`, input channel supports 1/3.
          Input shape should be limited from [4, 6] to [8192, 4096].

        Args:
            device_target (str, optional): The operator will be executed on this device. Currently supported
                values are ``"CPU"`` and ``"Ascend"`` . Default: ``"CPU"`` .

        Raises:
            TypeError: If `device_target` is not of type str.
            ValueError: If `device_target` is not within the valid set of ["CPU", "Ascend"].

        Supported Platforms:
            ``CPU`` ``Ascend``

        Examples:
            >>> import numpy as np
            >>> import mindspore.dataset as ds
            >>> import mindspore.dataset.vision as vision
            >>> from mindspore.dataset.vision import Inter
            >>>
            >>> # Use the transform in dataset pipeline mode
            >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
            >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
            >>> resize_op = vision.Resize([100, 75], Inter.BICUBIC)
            >>> transforms_list = [resize_op]
            >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
            >>> normalize_op = vision.Normalize(mean=[121.0, 115.0, 100.0], std=[70.0, 68.0, 71.0]).device("Ascend")
            >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=normalize_op, input_columns=["image"])
            >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            ...     print(item["image"].shape, item["image"].dtype)
            ...     break
            (100, 75, 3) float32
            >>>
            >>> # Use the transform in eager mode
            >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
            >>> output = vision.Normalize(mean=[121.0, 115.0, 100.0], std=[70.0, 68.0, 71.0]).device("Ascend")(data)
            >>> print(output.shape, output.dtype)
            (100, 100, 3) float32

        Tutorial Examples:
            - `Illustration of vision transforms
              <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
        """
        self.device_target = device_target
        return self

    def parse(self):
        return cde.NormalizeOperation(self.mean, self.std, self.is_hwc, self.device_target)


class NormalizePad(ImageTensorOperation):
    """
    Normalize the input image with respect to mean and standard deviation then pad an extra channel with value zero.

    Args:
        mean (sequence): List or tuple of mean values for each channel, with respect to channel order.
            The mean values must be in the range [0.0, 255.0].
        std (sequence): List or tuple of standard deviations for each channel, with respect to channel order.
            The standard deviation values must be in range (0.0, 255.0].
        dtype (str, optional): Set the output data type of the normalized image. Default: ``"float32"``.
        is_hwc (bool, optional): Specify the format of input image.
            ``True`` - HWC format, ``False`` - CHW format. Default: ``True``.

    Raises:
        TypeError: If `mean` is not of type sequence.
        TypeError: If `std` is not of type sequence.
        TypeError: If `dtype` is not of type string.
        TypeError: If `is_hwc` is not of type bool.
        ValueError: If `mean` is not in range [0.0, 255.0].
        ValueError: If `std` is not in range (0.0, 255.0].
        RuntimeError: If given tensor shape is not <H, W>, <H, W, C> or <C, H, W>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> normalize_pad_op = vision.NormalizePad(mean=[121.0, 115.0, 100.0],
        ...                                        std=[70.0, 68.0, 71.0],
        ...                                        dtype="float32")
        >>> transforms_list = [normalize_pad_op]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (100, 100, 4) float32
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
        >>> output = vision.NormalizePad(mean=[121.0, 115.0, 100.0], std=[70.0, 68.0, 71.0], dtype="float32")(data)
        >>> print(output.shape, output.dtype)
        (100, 100, 4) float32
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


class Pad(ImageTensorOperation, PyTensorOperation):
    """
    Pad the image according to padding parameters.

    Supports Ascend hardware acceleration and can be enabled through the `.device("Ascend")` method.

    Args:
        padding (Union[int, Sequence[int, int], Sequence[int, int, int, int]]): The number of pixels
            to pad each border of the image.

            - If a single number is provided, it pads all borders with this value.
            - If a tuple or list of 2 values are provided, it pads the (left and right)
              with the first value and (top and bottom) with the second value.
            - If 4 values are provided as a list or tuple, it pads the left, top, right and bottom respectively.

            The pad values must be non-negative.

        fill_value (Union[int, tuple[int]], optional): The pixel intensity of the borders, only valid for
            `padding_mode` ``Border.CONSTANT``.

            - If it is a 3-tuple, it is used to fill R, G, B channels respectively.
            - If it is an integer, it is used for all RGB channels.

            The fill_value values must be in range [0, 255]. Default: ``0``.
        padding_mode (Border, optional): The method of padding. It can be ``Border.CONSTANT``,
            ``Border.EDGE``, ``Border.REFLECT`` or ``Border.SYMMETRIC``. Default: ``Border.CONSTANT``.

            - ``Border.CONSTANT`` , means it fills the border with constant values.

            - ``Border.EDGE`` , means it pads with the last value on the edge.

            - ``Border.REFLECT`` , means it reflects the values on the edge omitting the last
              value on the edge.

            - ``Border.SYMMETRIC`` , means it reflects the values on the edge repeating the last
              value on the edge.

    Raises:
        TypeError: If `padding` is not of type int or Sequence[int, int], or Sequence[int, int, int, int].
        TypeError: If `fill_value` is not of type int or tuple[int].
        TypeError: If `padding_mode` is not of type :class:`mindspore.dataset.vision.Border` .
        ValueError: If `padding` is negative.
        ValueError: If `fill_value` is not in range [0, 255].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU`` ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> transforms_list = [vision.Pad([100, 100, 100, 100])]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (300, 300, 3) uint8
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
        >>> output = vision.Pad([100, 100, 100, 100])(data)
        >>> print(output.shape, output.dtype)
        (300, 300, 3) uint8

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """

    @check_pad
    def __init__(self, padding, fill_value=0, padding_mode=Border.CONSTANT):
        super().__init__()
        padding = parse_padding(padding)
        if isinstance(fill_value, int):
            fill_value = tuple([fill_value] * 3)
        self.padding = padding
        self.fill_value = fill_value
        self.random = False
        self.padding_mode = padding_mode

    @check_device_target
    def device(self, device_target="CPU"):
        """
        Set the device for the current operator execution.

        When the device is Ascend, input/output shape should be limited from [4, 6] to [32768, 32768].

        Args:
            device_target (str, optional): The operator will be executed on this device. Currently supports
                ``"CPU"`` and ``"Ascend"`` . Default: ``"CPU"`` .

        Raises:
            TypeError: If `device_target` is not of type str.
            ValueError: If `device_target` is not within the valid set of ["CPU", "Ascend"].

        Supported Platforms:
            ``CPU`` ``Ascend``

        Examples:
            >>> import numpy as np
            >>> import mindspore.dataset as ds
            >>> import mindspore.dataset.vision as vision
            >>>
            >>> # Use the transform in dataset pipeline mode
            >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
            >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
            >>> pad_op = vision.Pad([100, 100, 100, 100]).device("Ascend")
            >>> transforms_list = [pad_op]
            >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
            >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            ...     print(item["image"].shape, item["image"].dtype)
            ...     break
            (300, 300, 3) uint8
            >>>
            >>> # Use the transform in eager mode
            >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
            >>> output = vision.Pad([100, 100, 100, 100]).device("Ascend")(data)
            >>> print(output.shape, output.dtype)
            (300, 300, 3) uint8

        Tutorial Examples:
            - `Illustration of vision transforms
              <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
        """
        self.device_target = device_target
        return self

    def parse(self):
        return cde.PadOperation(self.padding, self.fill_value, Border.to_c_type(self.padding_mode), self.device_target)

    def _execute_py(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image, padded image.
        """
        return util.pad(img, self.padding, self.fill_value, Border.to_python_type(self.padding_mode))


class Perspective(ImageTensorOperation, PyTensorOperation):
    """
    Apply perspective transformation on input image.

    Supports Ascend hardware acceleration and can be enabled through the `.device("Ascend")` method.

    Args:
        start_points (Sequence[Sequence[int, int]]): Sequence of the starting point coordinates, containing four
            two-element subsequences, corresponding to [top-left, top-right, bottom-right, bottom-left] of the
            quadrilateral in the original image.
        end_points (Sequence[Sequence[int, int]]): Sequence of the ending point coordinates, containing four
            two-element subsequences, corresponding to [top-left, top-right, bottom-right, bottom-left] of the
            quadrilateral in the target image.
        interpolation (Inter, optional): Image interpolation method defined by :class:`~.vision.Inter` .
            Default: ``Inter.BILINEAR``.

            - Ascend Mode: Only ``Inter.LINEAR`` and ``Inter.NEAREST`` interpolation methods are supported
              when the device is set to Ascend by `.device("Ascend")` .

    Raises:
        TypeError: If `start_points` is not of type Sequence[Sequence[int, int]].
        TypeError: If `end_points` is not of type Sequence[Sequence[int, int]].
        TypeError: If `interpolation` is not of type :class:`~.vision.Inter` .
        RuntimeError: If shape of the input image is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU`` ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>> from mindspore.dataset.vision import Inter
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> start_points = [[0, 63], [63, 63], [63, 0], [0, 0]]
        >>> end_points = [[0, 32], [32, 32], [32, 0], [0, 0]]
        >>> transforms_list = [vision.Perspective(start_points, end_points, Inter.BILINEAR)]
        >>> # apply the transform to dataset through map function
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns="image")
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (100, 100, 3) uint8
        >>>
        >>> # Use the transform in eager mode
        >>> start_points = [[0, 63], [63, 63], [63, 0], [0, 0]]
        >>> end_points = [[0, 32], [32, 32], [32, 0], [0, 0]]
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
        >>> output = vision.Perspective(start_points, end_points, Inter.BILINEAR)(data)
        >>> print(output.shape, output.dtype)
        (100, 100, 3) uint8

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """

    @check_perspective
    def __init__(self, start_points, end_points, interpolation=Inter.BILINEAR):
        super().__init__()
        self.start_points = start_points
        self.end_points = end_points
        self.interpolation = interpolation
        if interpolation in [Inter.AREA, Inter.PILCUBIC]:
            self.implementation = Implementation.C
        elif interpolation == Inter.ANTIALIAS:
            self.implementation = Implementation.PY
        self.random = False

    @check_device_target
    def device(self, device_target="CPU"):
        """
        Set the device for the current operator execution.

        When the device is Ascend, input type supports `uint8` and `float32`,
        input channel supports 1 and 3. The input data has a height limit of [6, 8192]
        and a width limit of [10, 4096].

        Args:
            device_target (str, optional): The operator will be executed on this device. Currently supports
                ``"CPU"`` and ``"Ascend"`` . Default: ``"CPU"`` .

        Raises:
            TypeError: If `device_target` is not of type str.
            ValueError: If `device_target` is not within the valid set of ["CPU", "Ascend"].

        Supported Platforms:
            ``CPU`` ``Ascend``

        Examples:
            >>> import numpy as np
            >>> import mindspore.dataset as ds
            >>> import mindspore.dataset.vision as vision
            >>> from mindspore.dataset.vision import Inter
            >>>
            >>> # Use the transform in dataset pipeline mode
            >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
            >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
            >>> start_points = [[0, 63], [63, 63], [63, 0], [0, 0]]
            >>> end_points = [[0, 32], [32, 32], [32, 0], [0, 0]]
            >>> perspective_op = vision.Perspective(start_points, end_points).device("Ascend")
            >>> transforms_list = [perspective_op]
            >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
            >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            ...     print(item["image"].shape, item["image"].dtype)
            ...     break
            (100, 100, 3) uint8
            >>>
            >>> # Use the transform in eager mode
            >>> start_points = [[0, 63], [63, 63], [63, 0], [0, 0]]
            >>> end_points = [[0, 32], [32, 32], [32, 0], [0, 0]]
            >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
            >>> output = vision.Perspective(start_points, end_points, Inter.BILINEAR).device("Ascend")(data)
            >>> print(output.shape, output.dtype)
            (100, 100, 3) uint8

        Tutorial Examples:
            - `Illustration of vision transforms
              <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
        """
        self.device_target = device_target
        if self.interpolation not in [Inter.BILINEAR, Inter.NEAREST] and self.device_target == "Ascend":
            raise RuntimeError("Invalid interpolation mode, only support BILINEAR and NEAREST.")
        return self

    def parse(self):
        if self.interpolation == Inter.ANTIALIAS:
            raise TypeError("Current Interpolation is not supported with NumPy input.")
        return cde.PerspectiveOperation(self.start_points, self.end_points,
                                        Inter.to_c_type(self.interpolation), self.device_target)

    def _execute_py(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be perspectived.

        Returns:
            PIL Image, perspectived image.
        """
        if self.interpolation in [Inter.AREA, Inter.PILCUBIC]:
            raise TypeError("Current Interpolation is not supported with PIL input.")
        return util.perspective(img, self.start_points, self.end_points, Inter.to_python_type(self.interpolation))


class Posterize(ImageTensorOperation):
    """
    Reduce the bit depth of the color channels of image to create a high contrast and vivid color effect,
    similar to that seen in posters or printed materials.

    Supports Ascend hardware acceleration and can be enabled through the `.device("Ascend")` method.

    Args:
        bits (int): The number of bits to keep for each channel, should be in range [0, 8].

    Raises:
        TypeError: If `bits` is not of type int.
        ValueError: If `bits` is not in range [0, 8].
        RuntimeError: If shape of the input image is not <H, W> or <H, W, C>.

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> transforms_list = [vision.Posterize(4)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (100, 100, 3) uint8
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
        >>> output = vision.Posterize(4)(data)
        >>> print(output.shape, output.dtype)
        (100, 100, 3) uint8

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """

    @check_posterize
    def __init__(self, bits):
        super().__init__()
        self.bits = bits
        self.implementation = Implementation.C

    @check_device_target
    def device(self, device_target="CPU"):
        """
        Set the device for the current operator execution.

        When the device is Ascend, input type supports `uint8`/`float32`, input channel supports 1 and 3.
        The input data has a height limit of [4, 8192] and a width limit of [6, 4096].

        Args:
            device_target (str, optional): The operator will be executed on this device. Currently supports
                ``"CPU"`` and ``"Ascend"`` . Default: ``"CPU"`` .

        Raises:
            TypeError: If `device_target` is not of type str.
            ValueError: If `device_target` is not within the valid set of ["CPU", "Ascend"].

        Supported Platforms:
            ``CPU`` ``Ascend``

        Examples:
            >>> import numpy as np
            >>> import mindspore.dataset as ds
            >>> import mindspore.dataset.vision as vision
            >>>
            >>> # Use the transform in dataset pipeline mode
            >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
            >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
            >>> posterize_op = vision.Posterize(4).device("Ascend")
            >>> transforms_list = [posterize_op]
            >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
            >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            ...     print(item["image"].shape, item["image"].dtype)
            ...     break
            (100, 100, 3) uint8
            >>>
            >>> # Use the transform in eager mode
            >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
            >>> output = vision.Posterize(4).device("Ascend")(data)
            >>> print(output.shape, output.dtype)
            (100, 100, 3) uint8

        Tutorial Examples:
            - `Illustration of vision transforms
              <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
        """
        self.device_target = device_target
        return self

    def parse(self):
        return cde.PosterizeOperation(self.bits, self.device_target)


class RandAugment(ImageTensorOperation):
    """
    Apply RandAugment data augmentation method on the input image.

    Refer to `RandAugment: Learning Augmentation Strategies from Data <https://arxiv.org/pdf/1909.13719.pdf>`_ .

    Only support 3-channel RGB image.

    Args:
        num_ops (int, optional): Number of augmentation transformations to apply sequentially. Default: ``2``.
        magnitude (int, optional): Magnitude for all the transformations, must be smaller than
            `num_magnitude_bins`. Default: ``9``.
        num_magnitude_bins (int, optional): The number of different magnitude values,
            must be no less than 2. Default: ``31``.
        interpolation (Inter, optional): Image interpolation method defined by :class:`~.vision.Inter` .
            Default: ``Inter.NEAREST``.
        fill_value (Union[int, tuple[int, int, int]], optional): Pixel fill value for the area outside the
            transformed image, must be in range [0, 255]. Default: ``0``.
            If int is provided, pad all RGB channels with this value.
            If tuple[int, int, int] is provided, pad R, G, B channels respectively.

    Raises:
        TypeError: If `num_ops` is not of type int.
        ValueError: If `num_ops` is negative.
        TypeError: If `magnitude` is not of type int.
        ValueError: If `magnitude` is not positive.
        TypeError: If `num_magnitude_bins` is not of type int.
        ValueError: If `num_magnitude_bins` is less than 2.
        TypeError: If `interpolation` is not of type :class:`~.vision.Inter` .
        TypeError: If `fill_value` is not of type int or tuple[int, int, int].
        ValueError: If `fill_value` is not in range [0, 255].
        RuntimeError: If shape of the input image is not <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>> from mindspore.dataset.vision import Inter
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> transforms_list = [vision.RandAugment()]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (100, 100, 3) uint8
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
        >>> output = vision.RandAugment(interpolation=Inter.BILINEAR, fill_value=255)(data)
        >>> print(output.shape, output.dtype)
        (100, 100, 3) uint8

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """

    @check_rand_augment
    def __init__(self, num_ops=2, magnitude=9, num_magnitude_bins=31, interpolation=Inter.NEAREST, fill_value=0):
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        if isinstance(fill_value, int):
            fill_value = tuple([fill_value] * 3)
        self.fill_value = fill_value
        self.implementation = Implementation.C

    def parse(self):
        return cde.RandAugmentOperation(self.num_ops, self.magnitude, self.num_magnitude_bins,
                                        Inter.to_c_type(self.interpolation), self.fill_value)


class RandomColorAdjust(ImageTensorOperation, PyTensorOperation):
    """
    Randomly adjust the brightness, contrast, saturation, and hue of the input image.

    Note:
        This operation is executed on the CPU by default, but it is also supported
        to be executed on the GPU or Ascend via heterogeneous acceleration.

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
            If it is a float, the range will be [-hue, hue]. `hue` should be in range [0, 0.5].
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
        ``CPU`` ``GPU`` ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> transform_op = vision.RandomColorAdjust(brightness=(0.5, 1),
        ...                                         contrast=(0.4, 1),
        ...                                         saturation=(0.3, 1))
        >>> transforms_list = [transform_op]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (100, 100, 3) uint8
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
        >>> output = vision.RandomColorAdjust(brightness=(0.5, 1), contrast=(0.4, 1), saturation=(0.3, 1))(data)
        >>> print(output.shape, output.dtype)
        (100, 100, 3) uint8

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
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

    def _execute_py(self, img):
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


class RandomCrop(ImageTensorOperation, PyTensorOperation):
    """
    Crop the input image at a random location. If input image size is smaller than output size,
    input image will be padded before cropping.

    Note:
        If the input image is more than one, then make sure that the image size is the same.


    Args:
        size (Union[int, Sequence[int]]): The output size of the cropped image. The size value(s) must be positive.

            - If size is an integer, a square crop of size (size, size) is returned.
            - If size is a sequence of length 2, an image of size (height, width) will be cropped.

        padding (Union[int, Sequence[int]], optional): The number of pixels to pad each border of the image.
            The padding value(s) must be non-negative. Default: ``None``.

            - If `padding` is not ``None``, pad image first with padding values.
            - If a single number is provided, pad all borders with this value.
            - If a tuple or list of 2 values are provided, pad the (left and right)
              with the first value and (top and bottom) with the second value.
            - If 4 values are provided as a list or tuple,
              pad the left, top, right and bottom respectively.

        pad_if_needed (bool, optional): Pad the image if either side is smaller than
            the given output size. Default: ``False``.
        fill_value (Union[int, tuple[int]], optional): The pixel intensity of the borders, only valid for
            padding_mode ``Border.CONSTANT``. The fill_value values must be in range [0, 255]. Default: ``0``.

            - If it is a 3-tuple, it is used to fill R, G, B channels respectively.
            - If it is an integer, it is used for all RGB channels.

        padding_mode (Border, optional): The method of padding. Default: ``Border.CONSTANT``. It can be any of
            ``Border.CONSTANT``, ``Border.EDGE``, ``Border.REFLECT``, ``Border.SYMMETRIC``.

            - ``Border.CONSTANT`` , means it fills the border with constant values.

            - ``Border.EDGE`` , means it pads with the last value on the edge.

            - ``Border.REFLECT`` , means it reflects the values on the edge omitting the last
              value on the edge.

            - ``Border.SYMMETRIC`` , means it reflects the values on the edge repeating the last
              value on the edge.

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
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>> from mindspore.dataset.vision import Border
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> random_crop_op = vision.RandomCrop(64, [16, 16, 16, 16], padding_mode=Border.EDGE)
        >>> transforms_list = [random_crop_op]
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (64, 64, 3) uint8
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
        >>> output = vision.RandomCrop(8, [10, 10, 10, 10], padding_mode=Border.EDGE)(data)
        >>> print(output.shape, output.dtype)
        (8, 8, 3) uint8

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
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
        self.padding_mode = padding_mode

    def parse(self):
        return cde.RandomCropOperation(self.size, self.padding, self.pad_if_needed, self.fill_value,
                                       Border.to_c_type(self.padding_mode))

    def _execute_py(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be randomly cropped.

        Returns:
            PIL Image, cropped image.
        """
        return util.random_crop(img, self.size, self.padding, self.pad_if_needed,
                                self.fill_value, Border.to_python_type(self.padding_mode))


class RandomCropDecodeResize(ImageTensorOperation):
    """
    A combination of `Crop` , `Decode` and `Resize` . It will get better performance for JPEG images. This operation
    will crop the input image at a random location, decode the cropped image in RGB mode, and resize the decoded image.

    Args:
        size (Union[int, Sequence[int]]): The output size of the resized image. The size value(s) must be positive.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).
        scale (Union[list, tuple], optional): Range [min, max) of respective size of the
            original size to be cropped, which must be non-negative. Default: ``(0.08, 1.0)``.
        ratio (Union[list, tuple], optional): Range [min, max) of aspect ratio to be
            cropped, which must be non-negative. Default: ``(3. / 4., 4. / 3.)``.
        interpolation (Inter, optional): Image interpolation method defined by :class:`~.vision.Inter` .
            Default: ``Inter.BILINEAR``.
        max_attempts (int, optional): The maximum number of attempts to propose a valid crop_area. Default: ``10``.
            If exceeded, fall back to use center_crop instead. The `max_attempts` value must be positive.

    Raises:
        TypeError: If `size` is not of type int or Sequence[int].
        TypeError: If `scale` is not of type list or tuple.
        TypeError: If `ratio` is not of type list or tuple.
        TypeError: If `interpolation` is not of type :class:`~.vision.Inter` .
        TypeError: If `max_attempts` is not of type integer.
        ValueError: If `size` is not positive.
        ValueError: If `scale` is negative.
        ValueError: If `ratio` is negative.
        ValueError: If `max_attempts` is not positive.
        RuntimeError: If given tensor is not a 1D sequence.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import os
        >>> import numpy as np
        >>> from PIL import Image, ImageDraw
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>> from mindspore.dataset.vision import Inter
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> class MyDataset:
        ...     def __init__(self):
        ...         self.data = []
        ...         img = Image.new("RGB", (300, 300), (255, 255, 255))
        ...         draw = ImageDraw.Draw(img)
        ...         draw.ellipse(((0, 0), (100, 100)), fill=(255, 0, 0), outline=(255, 0, 0), width=5)
        ...         img.save("./1.jpg")
        ...         data = np.fromfile("./1.jpg", np.uint8)
        ...         self.data.append(data)
        ...
        ...     def __getitem__(self, index):
        ...         return self.data[0]
        ...
        ...     def __len__(self):
        ...         return 5
        >>>
        >>> my_dataset = MyDataset()
        >>> generator_dataset = ds.GeneratorDataset(my_dataset, column_names="image")
        >>> resize_crop_decode_op = vision.RandomCropDecodeResize(size=(50, 75),
        ...                                                       scale=(0.25, 0.5),
        ...                                                       interpolation=Inter.NEAREST,
        ...                                                       max_attempts=5)
        >>> transforms_list = [resize_crop_decode_op]
        >>> generator_dataset = generator_dataset.map(operations=transforms_list, input_columns=["image"])
        >>> for item in generator_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (50, 75, 3) uint8
        >>> os.remove("./1.jpg")
        >>>
        >>> # Use the transform in eager mode
        >>> img = Image.new("RGB", (300, 300), (255, 255, 255))
        >>> draw = ImageDraw.Draw(img)
        >>> draw.polygon([(50, 50), (150, 50), (100, 150)], fill=(0, 255, 0), outline=(0, 255, 0))
        >>> img.save("./2.jpg")
        >>> data = np.fromfile("./2.jpg", np.uint8)
        >>> output = vision.RandomCropDecodeResize(size=(50, 75), scale=(0, 10.0), ratio=(0.5, 0.5),
        ...                                        interpolation=Inter.BILINEAR, max_attempts=1)(data)
        >>> print(np.array(output).shape, np.array(output).dtype)
        (50, 75, 3) uint8
        >>> os.remove("./2.jpg")

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
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


class RandomCropWithBBox(ImageTensorOperation):
    """
    Crop the input image at a random location and adjust bounding boxes accordingly.

    Args:
        size (Union[int, Sequence[int]]): The output size of the cropped image. The size value(s) must be positive.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, an image of size (height, width) will be cropped.
        padding (Union[int, Sequence[int]], optional): The number of pixels to pad the image.
            The padding value(s) must be non-negative. Default: ``None``.
            If `padding` is not ``None``, first pad image with padding values.
            If a single number is provided, pad all borders with this value.
            If a tuple or list of 2 values are provided, pad the (left and right)
            with the first value and (top and bottom) with the second value.
            If 4 values are provided as a list or tuple, pad the left, top, right and bottom respectively.
        pad_if_needed (bool, optional): Pad the image if either side is smaller than
            the given output size. Default: ``False``.
        fill_value (Union[int, tuple[int]], optional): The pixel intensity of the borders, only valid for
            padding_mode ``Border.CONSTANT``. If it is a 3-tuple, it is used to fill R, G, B channels respectively.
            If it is an integer, it is used for all RGB channels.
            The fill_value values must be in range [0, 255]. Default: ``0``.
        padding_mode (Border, optional): The method of padding. Default: ``Border.CONSTANT``. It can be any of
            ``Border.CONSTANT``, ``Border.EDGE``, ``Border.REFLECT``, ``Border.SYMMETRIC``.

            - ``Border.CONSTANT`` , means it fills the border with constant values.

            - ``Border.EDGE`` , means it pads with the last value on the edge.

            - ``Border.REFLECT`` , means it reflects the values on the edge omitting the last
              value on the edge.

            - ``Border.SYMMETRIC`` , means it reflects the values on the edge repeating the last
              value on the edge.

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
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.float32)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> func = lambda img: (data, np.array([[0, 0, data.shape[1], data.shape[0]]]).astype(np.float32))
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=[func],
        ...                                                 input_columns=["image"],
        ...                                                 output_columns=["image", "bbox"])
        >>> random_crop_with_bbox_op = vision.RandomCropWithBBox([64, 64], [20, 20, 20, 20])
        >>> transforms_list = [random_crop_with_bbox_op]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image", "bbox"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     print(item["bbox"].shape, item["bbox"].dtype)
        ...     break
        (64, 64, 3) float32
        (1, 4) float32
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.float32)
        >>> func = lambda img: (data, np.array([[0, 0, data.shape[1], data.shape[0]]]).astype(data.dtype))
        >>> func_data, func_bboxes = func(data)
        >>> output = vision.RandomCropWithBBox([64, 64], [20, 20, 20, 20])(func_data, func_bboxes)
        >>> print(output[0].shape, output[0].dtype)
        (64, 64, 3) float32
        >>> print(output[1].shape, output[1].dtype)
        (1, 4) float32

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
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
        self.padding_mode = padding_mode
        self.implementation = Implementation.C

    def parse(self):
        border_type = Border.to_c_type(self.padding_mode)
        return cde.RandomCropWithBBoxOperation(self.size, self.padding, self.pad_if_needed, self.fill_value,
                                               border_type)


class RandomErasing(PyTensorOperation):
    """
    Randomly erase pixels within a random selected rectangle area on the input numpy.ndarray image.

    See `Random Erasing Data Augmentation <https://arxiv.org/pdf/1708.04896.pdf>`_ .

    Args:
        prob (float, optional): Probability of performing erasing, which
            must be in range [0.0, 1.0]. Default: ``0.5``.
        scale (Sequence[float, float], optional): Range of area scale of the erased area relative
            to the original image to select from, arranged in order of (min, max).
            Default: ``(0.02, 0.33)``.
        ratio (Sequence[float, float], optional): Range of aspect ratio of the erased area to select
            from, arranged in order of (min, max). Default: ``(0.3, 3.3)``.
        value (Union[int, str, Sequence[int, int, int]], optional): Pixel value used to pad the erased area.
            If a single integer is provided, it will be used for all RGB channels.
            If a sequence of length 3 is provided, it will be used for R, G, B channels respectively.
            If a string of ``'random'`` is provided, each pixel will be erased with a random value obtained
            from a standard normal distribution. Default: ``0``.
        inplace (bool, optional): Whether to apply erasing inplace. Default: ``False``.
        max_attempts (int, optional): The maximum number of attempts to propose a valid
            erased area, beyond which the original image will be returned. Default: ``10``.

    Raises:
        TypeError: If `prob` is not of type float.
        TypeError: If `scale` is not of type sequence.
        TypeError: If `ratio` is not of type sequence.
        TypeError: If `value` is not of type integer, string, or sequence.
        TypeError: If `inplace` is not of type boolean.
        TypeError: If `max_attempts` is not of type integer.
        ValueError: If `prob` is not in range [0.0, 1.0].
        ValueError: If `scale` is negative.
        ValueError: If `ratio` is negative.
        ValueError: If `value` is not in range [0, 255].
        ValueError: If `max_attempts` is not positive.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>> from mindspore.dataset.transforms import Compose
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> transforms_list = Compose([vision.ToTensor(),
        ...                            vision.RandomErasing(value='random')])
        >>> # apply the transform to dataset through map function
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns="image")
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (3, 100, 100) float32
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.random.randint(254, 255, size=(3, 100, 100)).astype(np.uint8)
        >>> output = vision.RandomErasing(prob=1.0, max_attempts=1)(data)
        >>> print(output.shape, output.dtype)
        (3, 100, 100) uint8

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
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

    def _execute_py(self, np_img):
        """
        Execute method.

        Args:
            np_img (numpy.ndarray): image in shape of <C, H, W> to be randomly erased.

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
        prob (float, optional): Probability of performing grayscale conversion,
            which must be in range [0.0, 1.0]. Default: ``0.1``.

    Raises:
        TypeError: If `prob` is not of type float.
        ValueError: If `prob` is not in range [0.0, 1.0].

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import os
        >>> import numpy as np
        >>> from PIL import Image, ImageDraw
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>> from mindspore.dataset.transforms import Compose
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> class MyDataset:
        ...     def __init__(self):
        ...         self.data = []
        ...         img = Image.new("RGB", (300, 300), (255, 255, 255))
        ...         draw = ImageDraw.Draw(img)
        ...         draw.ellipse(((0, 0), (100, 100)), fill=(255, 0, 0), outline=(255, 0, 0), width=5)
        ...         img.save("./1.jpg")
        ...         data = np.fromfile("./1.jpg", np.uint8)
        ...         self.data.append(data)
        ...
        ...     def __getitem__(self, index):
        ...         return self.data[0]
        ...
        ...     def __len__(self):
        ...         return 5
        >>>
        >>> my_dataset = MyDataset()
        >>> generator_dataset = ds.GeneratorDataset(my_dataset, column_names="image")
        >>> transforms_list = Compose([vision.Decode(to_pil=True),
        ...                            vision.RandomGrayscale(0.3),
        ...                            vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> generator_dataset = generator_dataset.map(operations=transforms_list, input_columns="image")
        >>> for item in generator_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (3, 300, 300) float32
        >>> os.remove("./1.jpg")
        >>>
        >>> # Use the transform in eager mode
        >>> img = Image.new("RGB", (300, 300), (255, 255, 255))
        >>> draw = ImageDraw.Draw(img)
        >>> draw.polygon([(50, 50), (150, 50), (100, 150)], fill=(0, 255, 0), outline=(0, 255, 0))
        >>> img.save("./2.jpg")
        >>> data = Image.open("./2.jpg")
        >>> output = vision.RandomGrayscale(1.0)(data)
        >>> print(np.array(output).shape, np.array(output).dtype)
        (300, 300, 3) uint8
        >>> os.remove("./2.jpg")

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """

    @check_prob
    def __init__(self, prob=0.1):
        super().__init__()
        self.prob = prob
        self.implementation = Implementation.PY

    def _execute_py(self, img):
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


class RandomHorizontalFlip(ImageTensorOperation, PyTensorOperation):
    """
    Randomly flip the input image horizontally with a given probability.

    Args:
        prob (float, optional): Probability of the image being flipped,
            which must be in range [0.0, 1.0]. Default: ``0.5``.

    Raises:
        TypeError: If `prob` is not of type float.
        ValueError: If `prob` is not in range [0.0, 1.0].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> transforms_list = [vision.RandomHorizontalFlip(0.75)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (100, 100, 3) uint8
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
        >>> output = vision.RandomHorizontalFlip(1.0)(data)
        >>> print(output.shape, output.dtype)
        (100, 100, 3) uint8

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """

    @check_prob
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def parse(self):
        return cde.RandomHorizontalFlipOperation(self.prob)

    def _execute_py(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be horizontally flipped.

        Returns:
            PIL Image, randomly horizontally flipped image.
        """
        return util.random_horizontal_flip(img, self.prob)


class RandomHorizontalFlipWithBBox(ImageTensorOperation):
    """
    Randomly flip the input image and its bounding box horizontally with a given probability.

    Args:
        prob (float, optional): Probability of the image being flipped,
            which must be in range [0.0, 1.0]. Default: ``0.5``.

    Raises:
        TypeError: If `prob` is not of type float.
        ValueError: If `prob` is not in range [0.0, 1.0].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.float32)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> func = lambda img: (data, np.array([[0, 0, data.shape[1], data.shape[0]]]).astype(np.float32))
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=[func],
        ...                                                 input_columns=["image"],
        ...                                                 output_columns=["image", "bbox"])
        >>> transforms_list = [vision.RandomHorizontalFlipWithBBox(0.70)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image", "bbox"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     print(item["bbox"].shape, item["bbox"].dtype)
        ...     break
        (100, 100, 3) float32
        (1, 4) float32
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.float32)
        >>> func = lambda img: (data, np.array([[0, 0, data.shape[1], data.shape[0]]]).astype(data.dtype))
        >>> func_data, func_bboxes = func(data)
        >>> output = vision.RandomHorizontalFlipWithBBox(1)(func_data, func_bboxes)
        >>> print(output[0].shape, output[0].dtype)
        (100, 100, 3) float32
        >>> print(output[1].shape, output[1].dtype)
        (1, 4) float32

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """

    @check_prob
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob
        self.implementation = Implementation.C

    def parse(self):
        return cde.RandomHorizontalFlipWithBBoxOperation(self.prob)


class RandomLighting(ImageTensorOperation, PyTensorOperation):
    """
    Add AlexNet-style PCA-based noise to an image. The eigenvalue and eigenvectors for AlexNet's PCA noise are
    calculated from the ImageNet dataset.

    Args:
        alpha (float, optional): Intensity of the image, which must be non-negative. Default: ``0.05``.

    Raises:
        TypeError: If `alpha` is not of type float.
        ValueError: If `alpha` is negative.
        RuntimeError: If given tensor shape is not <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> transforms_list = [vision.RandomLighting(0.1)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (100, 100, 3) uint8
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
        >>> output = vision.RandomLighting(0.1)(data)
        >>> print(output.shape, output.dtype)
        (100, 100, 3) uint8

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """

    @check_alpha
    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha

    def parse(self):
        return cde.RandomLightingOperation(self.alpha)

    def _execute_py(self, img):
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
        distortion_scale (float, optional): Scale of distortion, in range [0.0, 1.0]. Default: ``0.5``.
        prob (float, optional): Probability of performing perspective transformation, which
            must be in range [0.0, 1.0]. Default: ``0.5``.
        interpolation (Inter, optional): Image interpolation method defined by :class:`~.vision.Inter` .
            Default: ``Inter.BICUBIC``.

    Raises:
        TypeError: If `distortion_scale` is not of type float.
        TypeError: If `prob` is not of type float.
        TypeError: If `interpolation` is not of type :class:`~.vision.Inter` .
        ValueError: If `distortion_scale` is not in range [0.0, 1.0].
        ValueError: If `prob` is not in range [0.0, 1.0].

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import os
        >>> import numpy as np
        >>> from PIL import Image, ImageDraw
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>> from mindspore.dataset.transforms import Compose
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> class MyDataset:
        ...     def __init__(self):
        ...         self.data = []
        ...         img = Image.new("RGB", (300, 300), (255, 255, 255))
        ...         draw = ImageDraw.Draw(img)
        ...         draw.ellipse(((0, 0), (100, 100)), fill=(255, 0, 0), outline=(255, 0, 0), width=5)
        ...         img.save("./1.jpg")
        ...         data = np.fromfile("./1.jpg", np.uint8)
        ...         self.data.append(data)
        ...
        ...     def __getitem__(self, index):
        ...         return self.data[0]
        ...
        ...     def __len__(self):
        ...         return 5
        >>>
        >>> my_dataset = MyDataset()
        >>> generator_dataset = ds.GeneratorDataset(my_dataset, column_names="image")
        >>> transforms_list = Compose([vision.Decode(to_pil=True),
        ...                            vision.RandomPerspective(prob=0.1),
        ...                            vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> generator_dataset = generator_dataset.map(operations=transforms_list, input_columns="image")
        >>> for item in generator_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (3, 300, 300) float32
        >>> os.remove("./1.jpg")
        >>>
        >>> # Use the transform in eager mode
        >>> img = Image.new("RGB", (300, 300), (255, 255, 255))
        >>> draw = ImageDraw.Draw(img)
        >>> draw.polygon([(50, 50), (150, 50), (100, 150)], fill=(0, 255, 0), outline=(0, 255, 0))
        >>> img.save("./2.jpg")
        >>> data = Image.open("./2.jpg")
        >>> output = vision.RandomPerspective(prob=1.0)(data)
        >>> print(np.array(output).shape, np.array(output).dtype)
        (300, 300, 3) uint8
        >>> os.remove("./2.jpg")

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """

    @check_random_perspective
    def __init__(self, distortion_scale=0.5, prob=0.5, interpolation=Inter.BICUBIC):
        super().__init__()
        self.distortion_scale = distortion_scale
        self.prob = prob
        self.interpolation = interpolation
        self.implementation = Implementation.PY

    def _execute_py(self, img):
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
            return util.perspective(img, start_points, end_points, Inter.to_python_type(self.interpolation))
        return img


class RandomPosterize(ImageTensorOperation):
    """
    Reduce the number of bits for each color channel to posterize the input image randomly with a given probability.

    Args:
        bits (Union[int, Sequence[int]], optional): Range of random posterization to compress the image.
            Bits values must be in range [1, 8], and include at
            least one integer value in the given range. It must be in
            (min, max) or integer format. If min=max, then it is a single fixed
            magnitude operation. Default: ``(8, 8)``.

    Raises:
        TypeError: If `bits` is not of type integer or sequence of integer.
        ValueError: If `bits` is not in range [1, 8].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> transforms_list = [vision.RandomPosterize((6, 8))]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (100, 100, 3) uint8
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
        >>> output = vision.RandomPosterize(1)(data)
        >>> print(output.shape, output.dtype)
        (100, 100, 3) uint8

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """

    @check_random_posterize
    def __init__(self, bits=(8, 8)):
        super().__init__()
        self.bits = bits
        self.implementation = Implementation.C

    def parse(self):
        bits = self.bits
        if isinstance(bits, int):
            bits = (bits, bits)
        return cde.RandomPosterizeOperation(bits)


class RandomResizedCrop(ImageTensorOperation, PyTensorOperation):
    """
    This operation will crop the input image randomly,
    and resize the cropped image using a selected interpolation mode :class:`~.vision.Inter` .

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
        interpolation (Inter, optional): Image interpolation method defined by :class:`~.vision.Inter` .
            Default: ``Inter.BILINEAR``.
        max_attempts (int, optional): The maximum number of attempts to propose a valid
            crop_area. Default: ``10``. If exceeded, fall back to use center_crop instead.

    Raises:
        TypeError: If `size` is not of type int or Sequence[int].
        TypeError: If `scale` is not of type tuple or list.
        TypeError: If `ratio` is not of type tuple or list.
        TypeError: If `interpolation` is not of type :class:`~.vision.Inter` .
        TypeError: If `max_attempts` is not of type int.
        ValueError: If `size` is not positive.
        ValueError: If `scale` is negative.
        ValueError: If `ratio` is negative.
        ValueError: If `max_attempts` is not positive.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>> from mindspore.dataset.vision import Inter
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> resize_crop_op = vision.RandomResizedCrop(size=(50, 75), scale=(0.25, 0.5),
        ...                                           interpolation=Inter.BILINEAR)
        >>> transforms_list = [resize_crop_op]
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (50, 75, 3) uint8
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
        >>> output = vision.RandomResizedCrop(size=(50, 75), scale=(0.25, 0.5), interpolation=Inter.BILINEAR)(data)
        >>> print(output.shape, output.dtype)
        (50, 75, 3) uint8

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
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
        if interpolation in [Inter.AREA, Inter.PILCUBIC]:
            self.implementation = Implementation.C
        elif interpolation == Inter.ANTIALIAS:
            self.implementation = Implementation.PY
        self.max_attempts = max_attempts

    def parse(self):
        if self.interpolation == Inter.ANTIALIAS:
            raise TypeError("Current Interpolation is not supported with NumPy input.")
        return cde.RandomResizedCropOperation(self.size, self.scale, self.ratio, Inter.to_c_type(self.interpolation),
                                              self.max_attempts)

    def _execute_py(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be randomly cropped and resized.

        Returns:
            PIL Image, randomly cropped and resized image.
        """
        if self.interpolation in [Inter.AREA, Inter.PILCUBIC]:
            raise TypeError("Current Interpolation is not supported with PIL input.")
        return util.random_resize_crop(img, self.size, self.scale, self.ratio,
                                       Inter.to_python_type(self.interpolation), self.max_attempts)


class RandomResizedCropWithBBox(ImageTensorOperation):
    """
    Crop the input image to a random size and aspect ratio and adjust bounding boxes accordingly.

    Args:
        size (Union[int, Sequence[int]]): The size of the output image. The size value(s) must be positive.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).
        scale (Union[list, tuple], optional): Range (min, max) of respective size of the original
            size to be cropped, which must be non-negative. Default: ``(0.08, 1.0)``.
        ratio (Union[list, tuple], optional): Range (min, max) of aspect ratio to be
            cropped, which must be non-negative. Default: ``(3. / 4., 4. / 3.)``.
        interpolation (Inter, optional): Image interpolation method defined by :class:`~.vision.Inter` .
            Default: ``Inter.BILINEAR``.
        max_attempts (int, optional): The maximum number of attempts to propose a valid
            crop area. Default: ``10``. If exceeded, fall back to use center crop instead.

    Raises:
        TypeError: If `size` is not of type int or Sequence[int].
        TypeError: If `scale` is not of type list or tuple.
        TypeError: If `ratio` is not of type list or tuple.
        TypeError: If `interpolation` is not of type :class:`~.vision.Inter` .
        TypeError: If `max_attempts` is not of type integer.
        ValueError: If `size` is not positive.
        ValueError: If `scale` is negative.
        ValueError: If `ratio` is negative.
        ValueError: If `max_attempts` is not positive.
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>> from mindspore.dataset.vision import Inter
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.float32)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> func = lambda img: (data, np.array([[0, 0, data.shape[1], data.shape[0]]]).astype(np.float32))
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=[func],
        ...                                                 input_columns=["image"],
        ...                                                 output_columns=["image", "bbox"])
        >>> bbox_op = vision.RandomResizedCropWithBBox(size=50, interpolation=Inter.NEAREST)
        >>> transforms_list = [bbox_op]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list,
        ...                                                 input_columns=["image", "bbox"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     print(item["bbox"].shape, item["bbox"].dtype)
        ...     break
        (50, 50, 3) float32
        (1, 4) float32
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.float32)
        >>> func = lambda img: (data, np.array([[0, 0, data.shape[1], data.shape[0]]]).astype(data.dtype))
        >>> func_data, func_bboxes = func(data)
        >>> output = vision.RandomResizedCropWithBBox((16, 64), (0.5, 0.5), (0.5, 0.5))(func_data, func_bboxes)
        >>> print(output[0].shape, output[0].dtype)
        (16, 64, 3) float32
        >>> print(output[1].shape, output[1].dtype)
        (1, 4) float32

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
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


class RandomResizeWithBBox(ImageTensorOperation):
    """
    Tensor operation to resize the input image
    using a randomly selected interpolation mode :class:`~.vision.Inter` and to adjust
    bounding boxes accordingly.

    Args:
        size (Union[int, Sequence[int]]): The output size of the resized image. The size value(s) must be positive.
            If size is an integer, the smaller edge of the image will be resized to this value with
            the same image aspect ratio.
            If size is a sequence of length 2, it should be (height, width).

    Raises:
        TypeError: If `size` is not of type int or Sequence[int].
        ValueError: If `size` is not positive.
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import copy
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.float32)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> func = lambda img: (data, np.array([[0, 0, data.shape[1], data.shape[0]]]).astype(np.float32))
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=[func],
        ...                                                 input_columns=["image"],
        ...                                                 output_columns=["image", "bbox"])
        >>> numpy_slices_dataset2 = copy.deepcopy(numpy_slices_dataset)
        >>>
        >>> # 1) randomly resize image with bounding boxes, keeping aspect ratio
        >>> transforms_list1 = [vision.RandomResizeWithBBox(60)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list1,
        ...                                                 input_columns=["image", "bbox"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     print(item["bbox"].shape, item["bbox"].dtype)
        ...     break
        (60, 60, 3) float32
        (1, 4) float32
        >>>
        >>> # 2) randomly resize image with bounding boxes to portrait style
        >>> transforms_list2 = [vision.RandomResizeWithBBox((80, 60))]
        >>> numpy_slices_dataset2 = numpy_slices_dataset2.map(operations=transforms_list2,
        ...                                                   input_columns=["image", "bbox"])
        >>> for item in numpy_slices_dataset2.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     print(item["bbox"].shape, item["bbox"].dtype)
        ...     break
        (80, 60, 3) float32
        (1, 4) float32
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.float32)
        >>> func = lambda img: (data, np.array([[0, 0, data.shape[1], data.shape[0]]]).astype(data.dtype))
        >>> func_data, func_bboxes = func(data)
        >>> output = vision.RandomResizeWithBBox(64)(func_data, func_bboxes)
        >>> print(output[0].shape, output[0].dtype)
        (64, 64, 3) float32
        >>> print(output[1].shape, output[1].dtype)
        (1, 4) float32

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
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


class RandomSolarize(ImageTensorOperation):
    """
    Randomly selects a subrange within the specified threshold range and sets the pixel value within
    the subrange to (255 - pixel).

    Args:
        threshold (tuple, optional): Range of random solarize threshold. Default: ``(0, 255)``.
            Threshold values should always be in (min, max) format,
            where min and max are integers in the range [0, 255], and min <= max. The pixel values
            belonging to the [min, max] range will be inverted.
            If min=max, then invert all pixel values greater than or equal to min (or max).

    Raises:
        TypeError: If `threshold` is not of type tuple.
        ValueError: If `threshold` is not in range [0, 255].

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> transforms_list = [vision.RandomSolarize(threshold=(10,100))]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (100, 100, 3) uint8
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
        >>> output = vision.RandomSolarize(threshold=(1, 10))(data)
        >>> print(output.shape, output.dtype)
        (100, 100, 3) uint8

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """

    @check_random_solarize
    def __init__(self, threshold=(0, 255)):
        super().__init__()
        self.threshold = threshold
        self.implementation = Implementation.C

    def parse(self):
        return cde.RandomSolarizeOperation(self.threshold)


class RandomVerticalFlip(ImageTensorOperation, PyTensorOperation):
    """
    Randomly flip the input image vertically with a given probability.

    Args:
        prob (float, optional): Probability of the image being flipped, which
            must be in range [0.0, 1.0]. Default: ``0.5``.

    Raises:
        TypeError: If `prob` is not of type float.
        ValueError: If `prob` is not in range [0.0, 1.0].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> transforms_list = [vision.RandomVerticalFlip(0.25)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (100, 100, 3) uint8
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.uint8).reshape((2, 3))
        >>> output = vision.RandomVerticalFlip(1.0)(data)
        >>> print(output.shape, output.dtype)
        (2, 3) uint8

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """

    @check_prob
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def parse(self):
        return cde.RandomVerticalFlipOperation(self.prob)

    def _execute_py(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be vertically flipped.

        Returns:
            PIL Image, randomly vertically flipped image.
        """
        return util.random_vertical_flip(img, self.prob)


class RandomVerticalFlipWithBBox(ImageTensorOperation):
    """
    Flip the input image vertically, randomly with a given probability and adjust bounding boxes accordingly.

    Args:
        prob (float, optional): Probability of the image being flipped,
            which must be in range [0.0, 1.0]. Default: ``0.5``.

    Raises:
        TypeError: If `prob` is not of type float.
        ValueError: If `prob` is not in range [0.0, 1.0].
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.float32)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> func = lambda img: (data, np.array([[0, 0, data.shape[1], data.shape[0]]]).astype(np.float32))
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=[func],
        ...                                                 input_columns=["image"],
        ...                                                 output_columns=["image", "bbox"])
        >>> transforms_list = [vision.RandomVerticalFlipWithBBox(0.20)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image", "bbox"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     print(item["bbox"].shape, item["bbox"].dtype)
        ...     break
        (100, 100, 3) float32
        (1, 4) float32
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.float32)
        >>> func = lambda img: (data, np.array([[0, 0, data.shape[1], data.shape[0]]]).astype(data.dtype))
        >>> func_data, func_bboxes = func(data)
        >>> output = vision.RandomVerticalFlipWithBBox(1)(func_data, func_bboxes)
        >>> print(output[0].shape, output[0].dtype)
        (100, 100, 3) float32
        >>> print(output[1].shape, output[1].dtype)
        (1, 4) float32

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """

    @check_prob
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob
        self.implementation = Implementation.C

    def parse(self):
        return cde.RandomVerticalFlipWithBBoxOperation(self.prob)


class Rescale(ImageTensorOperation):
    """
    Rescale the input image with the given rescale and shift. This operation will rescale the input image
    with: output = image * rescale + shift.

    Note:
        This operation is executed on the CPU by default, but it is also supported
        to be executed on the GPU or Ascend via heterogeneous acceleration.

    Args:
        rescale (float): Rescale factor.
        shift (float): Shift factor.

    Raises:
        TypeError: If `rescale` is not of type float.
        TypeError: If `shift` is not of type float.

    Supported Platforms:
        ``CPU`` ``GPU`` ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> transforms_list = [vision.Rescale(1.0 / 255.0, -1.0)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (100, 100, 3) float32
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
        >>> output = vision.Rescale(1.0 / 255.0, -1.0)(data)
        >>> print(output.shape, output.dtype)
        (100, 100, 3) float32

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """

    @check_rescale
    def __init__(self, rescale, shift):
        super().__init__()
        self.rescale = rescale
        self.shift = shift
        self.implementation = Implementation.C

    def parse(self):
        return cde.RescaleOperation(self.rescale, self.shift)


class Resize(ImageTensorOperation, PyTensorOperation):
    """
    Resize the input image to the given size with a given interpolation mode :class:`~.vision.Inter` .

    Supports Ascend hardware acceleration and can be enabled through the `.device("Ascend")` method.

    Args:
        size (Union[int, Sequence[int]]): The output size of the resized image. The size value(s) must be positive.
            If size is an integer, the smaller edge of the image will be resized to this value with
            the same image aspect ratio.
            If size is a sequence of length 2, it should be (height, width).

            - CPU mode: If the execution device is CPU by ``.device("CPU")`` , the range of size is [1, 16777216].

            - Ascend mode: Set the executing device to Ascend by ``.device("Ascend")`` , the value of size will be in
              range [6, 32768].

        interpolation (Inter, optional): Image interpolation method defined by :class:`~.vision.Inter` .
            Default: ``Inter.LINEAR``.

            - Ascend mode: ``Inter.ANTIALIAS`` , ``Inter.AREA`` , ``Inter.PILCUBIC`` interpolation methods are not
              supported when the execution device is set to Ascend by ``.device("Ascend")`` .

    Raises:
        TypeError: If `size` is not of type int or Sequence[int].
        TypeError: If `interpolation` is not of type :class:`~.vision.Inter` .
        ValueError: If `size` is not positive.
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU`` ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>> from mindspore.dataset.vision import Inter
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> resize_op = vision.Resize([100, 75], Inter.BICUBIC)
        >>> transforms_list = [resize_op]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (100, 75, 3) uint8
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
        >>> output = vision.Resize([5, 5], Inter.BICUBIC)(data)
        >>> print(output.shape, output.dtype)
        (5, 5, 3) uint8

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """

    @check_resize_interpolation
    def __init__(self, size, interpolation=Inter.LINEAR):
        super().__init__()
        self.py_size = size
        if isinstance(size, int):
            size = (size,)
        self.c_size = size
        self.interpolation = interpolation
        if interpolation in [Inter.AREA, Inter.PILCUBIC]:
            self.implementation = Implementation.C
        elif interpolation == Inter.ANTIALIAS:
            self.implementation = Implementation.PY
        self.random = False

    @check_device_target
    def device(self, device_target="CPU"):
        """
        Set the device for the current operator execution.

        When the device is Ascend, input/output shape should be limited from [4, 6] to [32768, 32768].

        Args:
            device_target (str, optional): The operator will be executed on this device. Currently supports
                ``"CPU"`` and ``"Ascend"`` . Default: ``"CPU"`` .

        Raises:
            TypeError: If `device_target` is not of type str.
            ValueError: If `device_target` is not within the valid set of ["CPU", "Ascend"].

        Supported Platforms:
            ``CPU`` ``Ascend``

        Examples:
            >>> import numpy as np
            >>> import mindspore.dataset as ds
            >>> import mindspore.dataset.vision as vision
            >>> from mindspore.dataset.vision import Inter
            >>>
            >>> # Use the transform in dataset pipeline mode
            >>> resize_op = vision.Resize([100, 75], Inter.BICUBIC).device("Ascend")
            >>> transforms_list = [resize_op]
            >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
            >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
            >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
            >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            ...     print(item["image"].shape, item["image"].dtype)
            ...     break
            (100, 75, 3) uint8
            >>>
            >>> # Use the transform in eager mode
            >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
            >>> output = vision.Resize([25, 25], Inter.BICUBIC).device("Ascend")(data)
            >>> print(output.shape, output.dtype)
            (25, 25, 3) uint8

        Tutorial Examples:
            - `Illustration of vision transforms
              <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
        """
        self.device_target = device_target
        if self.interpolation not in [Inter.BILINEAR, Inter.CUBIC, Inter.NEAREST] and self.device_target == "Ascend":
            raise RuntimeError("Invalid interpolation mode, only support BILINEAR, CUBIC and NEAREST.")
        return self

    def parse(self):
        if self.interpolation == Inter.ANTIALIAS:
            raise TypeError("The current InterpolationMode is not supported with NumPy input.")
        return cde.ResizeOperation(self.c_size, Inter.to_c_type(self.interpolation), self.device_target)

    def _execute_py(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be resized.

        Returns:
            PIL Image, resized image.
        """
        if self.interpolation in [Inter.AREA, Inter.PILCUBIC]:
            raise TypeError("Current Interpolation is not supported with PIL input.")
        return util.resize(img, self.py_size, Inter.to_python_type(self.interpolation))


class ResizeWithBBox(ImageTensorOperation):
    """
    Resize the input image to the given size and adjust bounding boxes accordingly.

    Args:
        size (Union[int, Sequence[int]]): The output size of the resized image.
            If size is an integer, the smaller edge of the image will be resized to this value with
            the same image aspect ratio.
            If size is a sequence of length 2, it should be (height, width).
        interpolation (Inter, optional): Image interpolation method defined by :class:`~.vision.Inter` .
            Default: ``Inter.LINEAR``.

    Raises:
        TypeError: If `size` is not of type int or Sequence[int].
        TypeError: If `interpolation` is not of type :class:`~.vision.Inter` .
        ValueError: If `size` is not positive.
        RuntimeError: If given tensor shape is not <H, W> or <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>> from mindspore.dataset.vision import Inter
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.float32)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> func = lambda img: (data, np.array([[0, 0, data.shape[1], data.shape[0]]]).astype(np.float32))
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=[func],
        ...                                                 input_columns=["image"],
        ...                                                 output_columns=["image", "bbox"])
        >>> bbox_op = vision.ResizeWithBBox(50, Inter.NEAREST)
        >>> transforms_list = [bbox_op]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image", "bbox"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     print(item["bbox"].shape, item["bbox"].dtype)
        ...     break
        (50, 50, 3) float32
        (1, 4) float32
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.float32)
        >>> func = lambda img: (data, np.array([[0, 0, data.shape[1], data.shape[0]]]).astype(data.dtype))
        >>> func_data, func_bboxes = func(data)
        >>> output = vision.ResizeWithBBox(100)(func_data, func_bboxes)
        >>> print(output[0].shape, output[0].dtype)
        (100, 100, 3) float32
        >>> print(output[1].shape, output[1].dtype)
        (1, 4) float32

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
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


class Solarize(ImageTensorOperation):
    """
    Solarize the image by inverting all pixel values within the threshold.

    Supports Ascend hardware acceleration and can be enabled through the `.device("Ascend")` method.

    Args:
        threshold (Union[float, Sequence[float, float]]): Range of solarize threshold, should always
            be in (min, max) format, where min and max are integers in range [0, 255], and min <= max.
            The pixel values belonging to the [min, max] range will be inverted.
            If a single value is provided or min=max, then invert all pixel values greater than or
            equal to min (or max).

    Raises:
        TypeError: If `threshold` is not of type float or Sequence[float, float].
        ValueError: If `threshold` is not in range [0, 255].

    Supported Platforms:
        ``CPU`` ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> transforms_list = [vision.Solarize(threshold=(10, 100))]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (100, 100, 3) uint8
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
        >>> output = vision.Solarize(threshold=(1, 10))(data)
        >>> print(output.shape, output.dtype)
        (100, 100, 3) uint8

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """

    @check_solarize
    def __init__(self, threshold):
        super().__init__()
        if isinstance(threshold, (float, int)):
            threshold = (threshold, threshold)
        self.threshold = threshold
        self.implementation = Implementation.C

    @check_device_target
    def device(self, device_target="CPU"):
        """
        Set the device for the current operator execution.

        When the device is Ascend, input type only supports `uint8` , input channel supports 1 and 3.
        The input data has a height limit of [4, 8192] and a width limit of [6, 4096].

        Args:
            device_target (str, optional): The operator will be executed on this device. Currently supports
                ``"CPU"`` and ``"Ascend"`` . Default: ``"CPU"`` .

        Raises:
            TypeError: If `device_target` is not of type str.
            ValueError: If `device_target` is not within the valid set of ['CPU', 'Ascend'].

        Supported Platforms:
            ``CPU`` ``Ascend``

        Examples:
            >>> import numpy as np
            >>> import mindspore.dataset as ds
            >>> import mindspore.dataset.vision as vision
            >>>
            >>> # Use the transform in dataset pipeline mode
            >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
            >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
            >>> solarize_op = vision.Solarize(threshold=(10, 100)).device("Ascend")
            >>> transforms_list = [solarize_op]
            >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
            >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            ...     print(item["image"].shape, item["image"].dtype)
            ...     break
            (100, 100, 3) uint8
            >>>
            >>> # Use the transform in eager mode
            >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
            >>> output = vision.Solarize(threshold=(10, 100)).device("Ascend")(data)
            >>> print(output.shape, output.dtype)
            (100, 100, 3) uint8

        Tutorial Examples:
            - `Illustration of vision transforms
              <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
        """
        self.device_target = device_target
        return self

    def parse(self):
        return cde.SolarizeOperation(self.threshold, self.device_target)


class TenCrop(PyTensorOperation):
    """
    Crop the given image into one central crop and four corners with the flipped version of these.

    Args:
        size (Union[int, Sequence[int, int]]): The size of the cropped image.
            If a single integer is provided, a square of size (size, size) will be cropped with this value.
            If a sequence of length 2 is provided, an image of size (height, width) will be cropped.
        use_vertical_flip (bool, optional): If ``True``, flip the images vertically. Otherwise, flip them
            horizontally. Default: ``False``.

    Raises:
        TypeError: If `size` is not of type integer or sequence of integer.
        TypeError: If `use_vertical_flip` is not of type boolean.
        ValueError: If `size` is not positive.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import os
        >>> import numpy as np
        >>> from PIL import Image, ImageDraw
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>> from mindspore.dataset.transforms import Compose
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> class MyDataset:
        ...     def __init__(self):
        ...         self.data = []
        ...         img = Image.new("RGB", (300, 300), (255, 255, 255))
        ...         draw = ImageDraw.Draw(img)
        ...         draw.ellipse(((0, 0), (100, 100)), fill=(255, 0, 0), outline=(255, 0, 0), width=5)
        ...         img.save("./1.jpg")
        ...         data = np.fromfile("./1.jpg", np.uint8)
        ...         self.data.append(data)
        ...
        ...     def __getitem__(self, index):
        ...         return self.data[0]
        ...
        ...     def __len__(self):
        ...         return 5
        >>>
        >>> my_dataset = MyDataset()
        >>> generator_dataset = ds.GeneratorDataset(my_dataset, column_names="image")
        >>> transforms_list = Compose([vision.Decode(to_pil=True),
        ...                            vision.TenCrop(size=200),
        ...                            # 4D stack of 10 images
        ...                            lambda *images: np.stack([vision.ToTensor()(image) for image in images])])
        >>> # apply the transform to dataset through map function
        >>> generator_dataset = generator_dataset.map(operations=transforms_list, input_columns="image")
        >>> for item in generator_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (10, 3, 200, 200) float32
        >>> os.remove("./1.jpg")
        >>>
        >>> # Use the transform in eager mode
        >>> img = Image.new("RGB", (300, 300), (255, 255, 255))
        >>> draw = ImageDraw.Draw(img)
        >>> draw.polygon([(50, 50), (150, 50), (100, 150)], fill=(0, 255, 0), outline=(0, 255, 0))
        >>> img.save("./2.jpg")
        >>> data = Image.open("./2.jpg")
        >>> output = vision.TenCrop(size=200)(data)
        >>> print(len(output), np.array(output[0]).shape, np.array(output[0]).dtype)
        10 (200, 200, 3) uint8
        >>> os.remove("./2.jpg")

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
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

    def _execute_py(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            tuple, a tuple of 10 PIL Image, in order of top_left, top_right, bottom_left, bottom_right, center
                of the original image and top_left, top_right, bottom_left, bottom_right, center of the flipped image.
        """
        return util.ten_crop(img, self.size, self.use_vertical_flip)


class ToNumpy(PyTensorOperation):
    """
    Convert the PIL input image to numpy.ndarray image.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>> from mindspore.dataset.transforms import Compose
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> # Use ToNumpy to explicitly select C++ implementation of subsequent op
        >>> transforms_list = Compose([vision.RandomHorizontalFlip(0.5),
        ...                            vision.ToNumpy(),
        ...                            vision.Resize((50, 60))])
        >>> # apply the transform to dataset through map function
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns="image")
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (50, 60, 3) uint8
        >>>
        >>> # Use the transform in eager mode
        >>> data = list(np.random.randint(0, 255, size=(32, 32, 3, 3)).astype(np.int32))
        >>> output = vision.ToNumpy()(data)
        >>> print(type(output), output.shape, output.dtype)
        <class 'numpy.ndarray'> (32, 32, 3, 3) int32

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """

    def __init__(self):
        super().__init__()
        self.random = False
        # Use "Implementation.C" to indicate to select C++ implementation for next op in transforms list
        self.implementation = Implementation.C

    def _execute_py(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Image to be converted to numpy.ndarray.

        Returns:
            Image converted to numpy.ndarray
        """
        return np.array(img)


class ToPIL(PyTensorOperation):
    """
    Convert the input decoded numpy.ndarray image to PIL Image.

    Raises:
        TypeError: If the input image is not of type `numpy.ndarray` or `PIL.Image.Image` .

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>> from mindspore.dataset.transforms import Compose
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> # data is already decoded, but not in PIL Image format
        >>> transforms_list = Compose([vision.ToPIL(),
        ...                            vision.RandomHorizontalFlip(0.5),
        ...                            vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns="image")
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (3, 100, 100) float32
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
        >>> output = vision.ToPIL()(data)
        >>> print(type(output), np.array(output).shape, np.array(output).dtype)
        <class 'PIL.Image.Image'> (100, 100, 3) uint8

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """

    def __init__(self):
        super().__init__()
        self.random = False
        self.implementation = Implementation.PY

    def _execute_py(self, img):
        """
        Execute method.

        Args:
            img (numpy.ndarray): Decoded numpy.ndarray image to be converted to PIL Image.

        Returns:
            PIL Image, converted PIL Image.
        """
        return util.to_pil(img)


class ToTensor(ImageTensorOperation):
    """
    Convert the input PIL Image or numpy.ndarray to numpy.ndarray of the desired dtype, rescale the pixel value
    range from [0, 255] to [0.0, 1.0] and change the shape from <H, W, C> to <C, H, W>.

    Args:
        output_type (Union[mindspore.dtype, numpy.dtype], optional): The desired dtype of the output image.
            Default: ``np.float32`` .

    Raises:
        TypeError: If the input image is not of type `PIL.Image.Image` or `numpy.ndarray` .
        TypeError: If dimension of the input image is not 2 or 3.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>> from mindspore.dataset.transforms import Compose
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> # create a list of transformations to be applied to the "image" column of each data row
        >>> transforms_list = Compose([vision.RandomHorizontalFlip(0.5),
        ...                            vision.ToTensor()])
        >>> # apply the transform to dataset through map function
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns="image")
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (3, 100, 100) float32
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
        >>> output = vision.ToTensor()(data)
        >>> print(output.shape, output.dtype)
        (3, 100, 100) float32

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """

    @check_to_tensor
    def __init__(self, output_type=np.float32):
        super().__init__()
        if isinstance(output_type, typing.Type):
            output_type = mstype_to_detype(output_type)
        else:
            output_type = nptype_to_detype(output_type)
        self.output_type = str(output_type)
        self.random = False
        self.implementation = Implementation.C

    def parse(self):
        return cde.ToTensorOperation(self.output_type)

    def _execute_py(self, img):
        """
        Execute method.

        Args:
            img (PIL Image): Convert PIL Image to np.ndarray by method HWC to CHW and divide 255.

        Returns:
            Image converted to numpy.ndarray.
        """
        return util.to_tensor(img, self.output_type)


class ToType(TypeCast):
    """
    Cast the input to a given MindSpore data type or NumPy data type.

    It is the same as that of :class:`mindspore.dataset.transforms.TypeCast` .

    Note:
        This operation is executed on the CPU by default, but it is also supported
        to be executed on the GPU or Ascend via heterogeneous acceleration.

    Args:
        data_type (Union[mindspore.dtype, numpy.dtype]): The desired data type of the output image,
            such as ``numpy.float32`` .

    Raises:
        TypeError: If `data_type` is not of type :class:`mindspore.dtype` or `numpy.dtype` .

    Supported Platforms:
        ``CPU`` ``GPU`` ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>> import numpy as np
        >>> from mindspore.dataset.transforms import Compose
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> transforms_list = Compose([vision.RandomHorizontalFlip(0.5),
        ...                            vision.ToTensor(),
        ...                            vision.ToType(np.float32)])
        >>> # apply the transform to dataset through map function
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns="image")
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (3, 100, 100) float32
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.array([2.71606445312564e-03, 6.3476562564e-03]).astype(np.float64)
        >>> output = vision.ToType(np.float32)(data)
        >>> print(output, output.dtype)
        [0.00271606 0.00634766] float32

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """


class TrivialAugmentWide(ImageTensorOperation):
    """
    Apply the TrivialAugmentWide data augmentation method to the input image.

    Refer to
    `TrivialAugmentWide: Tuning-free Yet State-of-the-Art Data Augmentation <https://arxiv.org/abs/2103.10158>`_ .

    Only support 3-channel RGB image.

    Args:
        num_magnitude_bins (int, optional): The number of different magnitude values,
            must be greater than or equal to 2. Default: ``31``.
        interpolation (Inter, optional): Image interpolation method defined by :class:`~.vision.Inter` .
            Default: ``Inter.NEAREST``.
        fill_value (Union[int, tuple[int, int, int]], optional): Pixel fill value for the area outside the
            transformed image, must be in range of [0, 255]. Default: ``0``.
            If int is provided, pad all RGB channels with this value.
            If tuple[int, int, int] is provided, pad R, G, and B channels respectively.

    Raises:
        TypeError: If `num_magnitude_bins` is not of type int.
        ValueError: If `num_magnitude_bins` is less than 2.
        TypeError: If `interpolation` is not of type :class:`~.vision.Inter` .
        TypeError: If `fill_value` is not of type int or tuple[int, int, int].
        ValueError: If `fill_value` is not in range of [0, 255].
        RuntimeError: If the shape of the input image is not <H, W, C>.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.vision as vision
        >>> from mindspore.dataset.vision import Inter
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
        >>> transforms_list = [vision.TrivialAugmentWide(num_magnitude_bins=31,
        ...                                              interpolation=Inter.NEAREST,
        ...                                              fill_value=0)]
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
        >>> for item in numpy_slices_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["image"].shape, item["image"].dtype)
        ...     break
        (100, 100, 3) uint8
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
        >>> output = vision.TrivialAugmentWide()(data)
        >>> print(output.shape, output.dtype)
        (100, 100, 3) uint8

    Tutorial Examples:
        - `Illustration of vision transforms
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_
    """

    @check_trivial_augment_wide
    def __init__(self, num_magnitude_bins=31, interpolation=Inter.NEAREST, fill_value=0):
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        if isinstance(fill_value, int):
            fill_value = tuple([fill_value] * 3)
        self.fill_value = fill_value
        self.implementation = Implementation.C

    def parse(self):
        return cde.TrivialAugmentWideOperation(self.num_magnitude_bins, Inter.to_c_type(self.interpolation),
                                               self.fill_value)


def not_random(func):
    """
    Specify the function as "not random", i.e., it produces deterministic result.
    A Python function can only be cached after it is specified as "not random".
    """
    func.random = False
    return func
