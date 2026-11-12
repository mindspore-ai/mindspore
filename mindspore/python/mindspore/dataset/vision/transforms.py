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
import numpy as np
from PIL import Image

import mindspore._c_dataengine as cde
from mindspore._c_expression import typing
from . import py_transforms_util as util
from .utils import (
    Border,
    Inter,
    parse_padding,
)
from .validators import (
    check_center_crop,
    check_decode,
    check_normalize,
    check_prob,
    check_random_color_adjust,
    check_random_crop,
    check_random_resize_crop,
    check_range,
    check_rescale,
    check_resize_interpolation,
    check_to_tensor,
    check_device_target,
    FLOAT_MAX_INTEGER,
)
from ..core.datatypes import mstype_to_detype, nptype_to_detype
from ..transforms.py_transforms_util import Implementation
from ..transforms.transforms import (
    PyTensorOperation,
    TensorOperation,
)


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


def not_random(func):
    """
    Specify the function as "not random", i.e., it produces deterministic result.
    A Python function can only be cached after it is specified as "not random".
    """
    func.random = False
    return func
