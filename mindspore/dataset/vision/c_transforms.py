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
The module vision.c_transforms is inherited from _c_dataengine
and is implemented based on OpenCV in C++. It's a high performance module to
process images. Users can apply suitable augmentations on image data
to improve their training models.

.. Note::
    A constructor's arguments for every class in this module must be saved into the
    class attributes (self.xxx) to support save() and load().

Examples:
    >>> import mindspore.dataset as ds
    >>> import mindspore.dataset.transforms.c_transforms as c_transforms
    >>> import mindspore.dataset.vision.c_transforms as c_vision
    >>> from mindspore.dataset.vision import Border, Inter
    >>>
    >>> dataset_dir = "path/to/imagefolder_directory"
    >>> # create a dataset that reads all files in dataset_dir with 8 threads
    >>> data1 = ds.ImageFolderDataset(dataset_dir, num_parallel_workers=8)
    >>> # create a list of transformations to be applied to the image data
    >>> transforms_list = [c_vision.Decode(),
    >>>                    c_vision.Resize((256, 256), interpolation=Inter.LINEAR),
    >>>                    c_vision.RandomCrop(200, padding_mode=Border.EDGE),
    >>>                    c_vision.RandomRotation((0, 15)),
    >>>                    c_vision.Normalize((100, 115.0, 121.0), (71.0, 68.0, 70.0)),
    >>>                    c_vision.HWC2CHW()]
    >>> onehot_op = c_transforms.OneHot(num_classes=10)
    >>> # apply the transformation to the dataset through data1.map()
    >>> data1 = data1.map(operations=transforms_list, input_columns="image")
    >>> data1 = data1.map(operations=onehot_op, input_columns="label")
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

DE_C_INTER_MODE = {Inter.NEAREST: cde.InterpolationMode.DE_INTER_NEAREST_NEIGHBOUR,
                   Inter.LINEAR: cde.InterpolationMode.DE_INTER_LINEAR,
                   Inter.CUBIC: cde.InterpolationMode.DE_INTER_CUBIC,
                   Inter.AREA: cde.InterpolationMode.DE_INTER_AREA}

DE_C_BORDER_TYPE = {Border.CONSTANT: cde.BorderType.DE_BORDER_CONSTANT,
                    Border.EDGE: cde.BorderType.DE_BORDER_EDGE,
                    Border.REFLECT: cde.BorderType.DE_BORDER_REFLECT,
                    Border.SYMMETRIC: cde.BorderType.DE_BORDER_SYMMETRIC}

DE_C_IMAGE_BATCH_FORMAT = {ImageBatchFormat.NHWC: cde.ImageBatchFormat.DE_IMAGE_BATCH_FORMAT_NHWC,
                           ImageBatchFormat.NCHW: cde.ImageBatchFormat.DE_IMAGE_BATCH_FORMAT_NCHW}


def parse_padding(padding):
    if isinstance(padding, numbers.Number):
        padding = [padding] * 4
    if len(padding) == 2:
        left = right = padding[0]
        top = bottom = padding[1]
        padding = (left, top, right, bottom,)
    if isinstance(padding, list):
        padding = tuple(padding)
    return padding


class AutoContrast(cde.AutoContrastOp):
    """
    Apply automatic contrast on input image.

    Args:
        cutoff (float, optional): Percent of pixels to cut off from the histogram (default=0.0).
        ignore (Union[int, sequence], optional): Pixel values to ignore (default=None).

    Examples:
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>>
        >>> transforms_list = [c_vision.Decode(), c_vision.AutoContrast(cutoff=10.0, ignore=[10, 20])]
        >>> data1 = data1.map(operations=transforms_list, input_columns=["image"])
    """

    @check_auto_contrast
    def __init__(self, cutoff=0.0, ignore=None):
        if ignore is None:
            ignore = []
        if isinstance(ignore, int):
            ignore = [ignore]
        super().__init__(cutoff, ignore)


class RandomSharpness(cde.RandomSharpnessOp):
    """
    Adjust the sharpness of the input image by a fixed or random degree. Degree of 0.0 gives a blurred image,
    degree of 1.0 gives the original image, and degree of 2.0 gives a sharpened image.

    Args:
        degrees (tuple, optional): Range of random sharpness adjustment degrees. It should be in (min, max) format.
            If min=max, then it is a single fixed magnitude operation (default = (0.1, 1.9)).

    Raises:
        TypeError : If degrees is not a list or tuple.
        ValueError: If degrees is negative.
        ValueError: If degrees is in (max, min) format instead of (min, max).

    Examples:
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>>
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomSharpness(degrees=(0.2, 1.9))]
        >>> data1 = data1.map(operations=transforms_list, input_columns=["image"])
    """

    @check_positive_degrees
    def __init__(self, degrees=(0.1, 1.9)):
        self.degrees = degrees
        super().__init__(*degrees)


class Equalize(cde.EqualizeOp):
    """
    Apply histogram equalization on input image.

    Examples:
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>>
        >>> transforms_list = [c_vision.Decode(), c_vision.Equalize()]
        >>> data1 = data1.map(operations=transforms_list, input_columns=["image"])
    """


class Invert(cde.InvertOp):
    """
    Apply invert on input image in RGB mode.

    Examples:
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>>
        >>> transforms_list = [c_vision.Decode(), c_vision.Invert()]
        >>> data1 = data1.map(operations=transforms_list, input_columns=["image"])
    """


class Decode(cde.DecodeOp):
    """
    Decode the input image in RGB mode.

    Args:
        rgb (bool, optional): Mode of decoding input image (default=True).
            If True means format of decoded image is RGB else BGR(deprecated).

    Examples:
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>>
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomHorizontalFlip()]
        >>> data1 = data1.map(operations=transforms_list, input_columns=["image"])
    """

    def __init__(self, rgb=True):
        self.rgb = rgb
        super().__init__(self.rgb)

    def __call__(self, img):
        """
        Call method.

        Args:
            img (NumPy): Image to be decoded.

        Returns:
            img (NumPy), Decoded image.
        """
        if not isinstance(img, np.ndarray) or img.ndim != 1 or img.dtype.type is np.str_:
            raise TypeError("Input should be an encoded image with 1-D NumPy type, got {}.".format(type(img)))
        decode = cde.Execute(cde.DecodeOp(self.rgb))
        img = decode(cde.Tensor(np.asarray(img)))
        return img.as_array()


class CutMixBatch(cde.CutMixBatchOp):
    """
    Apply CutMix transformation on input batch of images and labels.
    Note that you need to make labels into one-hot format and batch before calling this function.

    Args:
        image_batch_format (Image Batch Format): The method of padding. Can be any of
            [ImageBatchFormat.NHWC, ImageBatchFormat.NCHW]
        alpha (float, optional): hyperparameter of beta distribution (default = 1.0).
        prob (float, optional): The probability by which CutMix is applied to each image (default = 1.0).

    Examples:
        >>> import mindspore.dataset.transforms.c_transforms as c_transforms
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>> from mindspore.dataset.transforms.vision import ImageBatchFormat
        >>>
        >>> onehot_op = c_transforms.OneHot(num_classes=10)
        >>> data1 = data1.map(operations=onehot_op, input_columns=["label"])
        >>> cutmix_batch_op = c_vision.CutMixBatch(ImageBatchFormat.NHWC, 1.0, 0.5)
        >>> data1 = data1.batch(5)
        >>> data1 = data1.map(operations=cutmix_batch_op, input_columns=["image", "label"])
    """

    @check_cut_mix_batch_c
    def __init__(self, image_batch_format, alpha=1.0, prob=1.0):
        self.image_batch_format = image_batch_format.value
        self.alpha = alpha
        self.prob = prob
        super().__init__(DE_C_IMAGE_BATCH_FORMAT[image_batch_format], alpha, prob)


class CutOut(cde.CutOutOp):
    """
    Randomly cut (mask) out a given number of square patches from the input NumPy image array.

    Args:
        length (int): The side length of each square patch.
        num_patches (int, optional): Number of patches to be cut out of an image (default=1).

    Examples:
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>>
        >>> transforms_list = [c_vision.Decode(), c_vision.CutOut(80, num_patches=10)]
        >>> data1 = data1.map(operations=transforms_list, input_columns=["image"])
    """

    @check_cutout
    def __init__(self, length, num_patches=1):
        self.length = length
        self.num_patches = num_patches
        fill_value = (0, 0, 0)
        super().__init__(length, length, num_patches, False, *fill_value)


class MixUpBatch(cde.MixUpBatchOp):
    """
    Apply MixUp transformation on input batch of images and labels. Each image is multiplied by a random weight (lambda)
    and then added to a randomly selected image from the batch multiplied by (1 - lambda). The same formula is also
    applied to the one-hot labels.
    Note that you need to make labels into one-hot format and batch before calling this function.

    Args:
        alpha (float, optional): Hyperparameter of beta distribution (default = 1.0).

    Examples:
        >>> import mindspore.dataset.transforms.c_transforms as c_transforms
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>>
        >>> onehot_op = c_transforms.OneHot(num_classes=10)
        >>> data1 = data1.map(operations=onehot_op, input_columns=["label"])
        >>> mixup_batch_op = c_vision.MixUpBatch(alpha=0.9)
        >>> data1 = data1.batch(5)
        >>> data1 = data1.map(operations=mixup_batch_op, input_columns=["image", "label"])
    """

    @check_mix_up_batch_c
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        super().__init__(alpha)


class Normalize(cde.NormalizeOp):
    """
    Normalize the input image with respect to mean and standard deviation.

    Args:
        mean (sequence): List or tuple of mean values for each channel, with respect to channel order.
            The mean values must be in range [0.0, 255.0].
        std (sequence): List or tuple of standard deviations for each channel, with respect to channel order.
            The standard deviation values must be in range (0.0, 255.0].

    Examples:
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>>
        >>> decode_op = c_vision.Decode()
        >>> normalize_op = c_vision.Normalize(mean=[121.0, 115.0, 100.0], std=[70.0, 68.0, 71.0])
        >>> transforms_list = [decode_op, normalize_op]
        >>> data1 = data1.map(operations=transforms_list, input_columns=["image"])
    """

    @check_normalize_c
    def __init__(self, mean, std):
        if len(mean) == 1:
            mean = [mean[0]] * 3
        if len(std) == 1:
            std = [std[0]] * 3
        self.mean = mean
        self.std = std
        super().__init__(*mean, *std)

    def __call__(self, img):
        """
        Call method.

        Args:
            img (NumPy or PIL image): Image array to be normalized.

        Returns:
            img (NumPy), Normalized Image array.
        """
        if not isinstance(img, (np.ndarray, Image.Image)):
            raise TypeError("Input should be NumPy or PIL image, got {}.".format(type(img)))
        normalize = cde.Execute(cde.NormalizeOp(*self.mean, *self.std))
        img = normalize(cde.Tensor(np.asarray(img)))
        return img.as_array()


class NormalizePad(cde.NormalizePadOp):
    """
    Normalize the input image with respect to mean and standard deviation then pad an extra channel with value zero.

    Args:
        mean (sequence): List or tuple of mean values for each channel, with respect to channel order.
            The mean values must be in range (0.0, 255.0].
        std (sequence): List or tuple of standard deviations for each channel, with respect to channel order.
            The standard deviation values must be in range (0.0, 255.0].
        dtype (str): Set the output data type of normalized image (default is "float32").

    Examples:
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>>
        >>> decode_op = c_vision.Decode()
        >>> normalize_op = c_vision.NormalizePad(mean=[121.0, 115.0, 100.0], std=[70.0, 68.0, 71.0], dtype="float32")
        >>> transforms_list = [decode_op, normalize_pad_op]
        >>> data1 = data1.map(operations=transforms_list, input_columns=["image"])
    """

    @check_normalizepad_c
    def __init__(self, mean, std, dtype="float32"):
        self.mean = mean
        self.std = std
        self.dtype = dtype
        super().__init__(*mean, *std, dtype)

    def __call__(self, img):
        """
        Call method.

        Args:
            img (NumPy or PIL image): Image array to be normalizepad.

        Returns:
            img (NumPy), NormalizePaded Image array.
        """
        if not isinstance(img, (np.ndarray, Image.Image)):
            raise TypeError("Input should be NumPy or PIL image, got {}.".format(type(img)))
        normalize_pad = cde.Execute(cde.NormalizePadOp(*self.mean, *self.std, self.dtype))
        img = normalize_pad(cde.Tensor(np.asarray(img)))
        return img.as_array()


class RandomAffine(cde.RandomAffineOp):
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
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>> from mindspore.dataset.transforms.vision import Inter
        >>>
        >>> decode_op = c_vision.Decode()
        >>> random_affine_op = c_vision.RandomAffine(degrees=15, translate=(-0.1, 0.1, 0, 0), scale=(0.9, 1.1),
        >>>     resample=Inter.NEAREST)
        >>> transforms_list = [decode_op, random_affine_op]
        >>> data1 = data1.map(operations=transforms_list, input_columns=["image"])
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

        super().__init__(degrees, translate, scale, shear, DE_C_INTER_MODE[resample], fill_value)


class RandomCrop(cde.RandomCropOp):
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
        fill_value (Union[int, tuple], optional): The pixel intensity of the borders if
            the padding_mode is Border.CONSTANT (default=0). If it is a 3-tuple, it is used to
            fill R, G, B channels respectively.
        padding_mode (Border mode, optional): The method of padding (default=Border.CONSTANT). It can be any of
            [Border.CONSTANT, Border.EDGE, Border.REFLECT, Border.SYMMETRIC].

            - Border.CONSTANT, means it fills the border with constant values.

            - Border.EDGE, means it pads with the last value on the edge.

            - Border.REFLECT, means it reflects the values on the edge omitting the last
              value of edge.

            - Border.SYMMETRIC, means it reflects the values on the edge repeating the last
              value of edge.

    Examples:
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>>
        >>> decode_op = c_vision.Decode()
        >>> random_crop_op = c_vision.RandomCrop(512, [200, 200, 200, 200], padding_mode=Border.EDGE)
        >>> transforms_list = [decode_op, random_crop_op]
        >>> data1 = data1.map(operations=transforms_list, input_columns=["image"])
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
        border_type = DE_C_BORDER_TYPE[padding_mode]

        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill_value = fill_value
        self.padding_mode = padding_mode.value

        super().__init__(*size, *padding, border_type, pad_if_needed, *fill_value)


class RandomCropWithBBox(cde.RandomCropWithBBoxOp):
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
        fill_value (Union[int, tuple], optional): The pixel intensity of the borders if
            the padding_mode is Border.CONSTANT (default=0). If it is a 3-tuple, it is used to
            fill R, G, B channels respectively.
        padding_mode (Border mode, optional): The method of padding (default=Border.CONSTANT). It can be any of
            [Border.CONSTANT, Border.EDGE, Border.REFLECT, Border.SYMMETRIC].

            - Border.CONSTANT, means it fills the border with constant values.

            - Border.EDGE, means it pads with the last value on the edge.

            - Border.REFLECT, means it reflects the values on the edge omitting the last
              value of edge.

            - Border.SYMMETRIC, means it reflects the values on the edge repeating the last
              value of edge.

    Examples:
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>>
        >>> decode_op = c_vision.Decode()
        >>> random_crop_with_bbox_op = c_vision.RandomCrop([512, 512], [200, 200, 200, 200])
        >>> transforms_list = [decode_op, random_crop_with_bbox_op]
        >>> data3 = data3.map(operations=transforms_list, input_columns=["image"])
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
        border_type = DE_C_BORDER_TYPE[padding_mode]

        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill_value = fill_value
        self.padding_mode = padding_mode.value

        super().__init__(*size, *padding, border_type, pad_if_needed, *fill_value)


class RandomHorizontalFlip(cde.RandomHorizontalFlipOp):
    """
    Flip the input image horizontally, randomly with a given probability.

    Args:
        prob (float, optional): Probability of the image being flipped (default=0.5).

    Examples:
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>>
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomHorizontalFlip(0.75)]
        >>> data1 = data1.map(operations=transforms_list, input_columns=["image"])
    """

    @check_prob
    def __init__(self, prob=0.5):
        self.prob = prob
        super().__init__(prob)


class RandomHorizontalFlipWithBBox(cde.RandomHorizontalFlipWithBBoxOp):
    """
    Flip the input image horizontally, randomly with a given probability and adjust bounding boxes accordingly.

    Args:
        prob (float, optional): Probability of the image being flipped (default=0.5).

    Examples:
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>>
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomHorizontalFlipWithBBox(0.70)]
        >>> data1 = data1.map(operations=transforms_list, input_columns=["image"])
    """

    @check_prob
    def __init__(self, prob=0.5):
        self.prob = prob
        super().__init__(prob)


class RandomPosterize(cde.RandomPosterizeOp):
    """
    Reduce the number of bits for each color channel.

    Args:
        bits (sequence or int, optional): Range of random posterize to compress image.
            Bits values must be in range of [1,8], and include at
            least one integer value in the given range. It must be in
            (min, max) or integer format. If min=max, then it is a single fixed
            magnitude operation (default=(8, 8)).

    Examples:
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>>
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomPosterize((6, 8))]
        >>> data1 = data1.map(operations=transforms_list, input_columns=["image"])
    """

    @check_posterize
    def __init__(self, bits=(8, 8)):
        self.bits = bits
        if isinstance(bits, int):
            bits = (bits, bits)
        super().__init__(bits)


class RandomVerticalFlip(cde.RandomVerticalFlipOp):
    """
    Flip the input image vertically, randomly with a given probability.

    Args:
        prob (float, optional): Probability of the image being flipped (default=0.5).

    Examples:
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>>
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomVerticalFlip(0.25)]
        >>> data1 = data1.map(operations=transforms_list, input_columns=["image"])
    """

    @check_prob
    def __init__(self, prob=0.5):
        self.prob = prob
        super().__init__(prob)


class RandomVerticalFlipWithBBox(cde.RandomVerticalFlipWithBBoxOp):
    """
    Flip the input image vertically, randomly with a given probability and adjust bounding boxes accordingly.

    Args:
        prob (float, optional): Probability of the image being flipped (default=0.5).

    Examples:
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>>
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomVerticalFlipWithBBox(0.20)]
        >>> data1 = data1.map(operations=transforms_list, input_columns=["image"])
    """

    @check_prob
    def __init__(self, prob=0.5):
        self.prob = prob
        super().__init__(prob)


class BoundingBoxAugment(cde.BoundingBoxAugmentOp):
    """
    Apply a given image transform on a random selection of bounding box regions of a given image.

    Args:
        transform: C++ transformation function to be applied on random selection
            of bounding box regions of a given image.
        ratio (float, optional): Ratio of bounding boxes to apply augmentation on.
            Range: [0, 1] (default=0.3).

    Examples:
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>>
        >>> # set bounding box operation with ratio of 1 to apply rotation on all bounding boxes
        >>> bbox_aug_op = c_vision.BoundingBoxAugment(c_vision.RandomRotation(90), 1)
        >>> # map to apply ops
        >>> data3 = data3.map(operations=[bbox_aug_op],
        >>>                   input_columns=["image", "bbox"],
        >>>                   output_columns=["image", "bbox"],
        >>>                   column_order=["image", "bbox"])
    """

    @check_bounding_box_augment_cpp
    def __init__(self, transform, ratio=0.3):
        self.ratio = ratio
        self.transform = transform
        super().__init__(transform, ratio)


class Resize(cde.ResizeOp):
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
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>> from mindspore.dataset.transforms.vision import Inter
        >>>
        >>> decode_op = c_vision.Decode()
        >>> resize_op = c_vision.Resize([100, 75], Inter.BICUBIC)
        >>> transforms_list = [decode_op, resize_op]
        >>> data1 = data1.map(operations=transforms_list, input_columns=["image"])
    """

    @check_resize_interpolation
    def __init__(self, size, interpolation=Inter.LINEAR):
        if isinstance(size, int):
            size = (size, 0)
        self.size = size
        self.interpolation = interpolation
        interpoltn = DE_C_INTER_MODE[interpolation]
        super().__init__(*size, interpoltn)

    def __call__(self, img):
        """
        Call method.

        Args:
            img (NumPy or PIL image): Image to be resized.

        Returns:
            img (NumPy), Resized image.
        """
        if not isinstance(img, (np.ndarray, Image.Image)):
            raise TypeError("Input should be NumPy or PIL image, got {}.".format(type(img)))
        resize = cde.Execute(cde.ResizeOp(*self.size, DE_C_INTER_MODE[self.interpolation]))
        img = resize(cde.Tensor(np.asarray(img)))
        return img.as_array()


class ResizeWithBBox(cde.ResizeWithBBoxOp):
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
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>> from mindspore.dataset.transforms.vision import Inter
        >>>
        >>> decode_op = c_vision.Decode()
        >>> bbox_op = c_vision.ResizeWithBBox(50, Inter.NEAREST)
        >>> transforms_list = [decode_op, bbox_op]
        >>> data3 = data3.map(operations=transforms_list, input_columns=["image"])
    """

    @check_resize_interpolation
    def __init__(self, size, interpolation=Inter.LINEAR):
        self.size = size
        self.interpolation = interpolation
        interpoltn = DE_C_INTER_MODE[interpolation]
        if isinstance(size, int):
            size = (size, 0)
        super().__init__(*size, interpoltn)


class RandomResizedCropWithBBox(cde.RandomCropAndResizeWithBBoxOp):
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
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>> from mindspore.dataset.transforms.vision import Inter
        >>>
        >>> decode_op = c_vision.Decode()
        >>> bbox_op = c_vision.RandomResizedCropWithBBox(size=50, interpolation=Inter.NEAREST)
        >>> transforms_list = [decode_op, bbox_op]
        >>> data3 = data3.map(operations=transforms_list, input_columns=["image"])
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
        interpoltn = DE_C_INTER_MODE[interpolation]
        super().__init__(*size, *scale, *ratio, interpoltn, max_attempts)


class RandomResizedCrop(cde.RandomCropAndResizeOp):
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
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>> from mindspore.dataset.transforms.vision import Inter
        >>>
        >>> decode_op = c_vision.Decode()
        >>> resize_crop_op = c_vision.RandomResizedCrop(size=(50, 75), scale=(0.25, 0.5),
        >>>     interpolation=Inter.BILINEAR)
        >>> transforms_list = [decode_op, resize_crop_op]
        >>> data1 = data1.map(operations=transforms_list, input_columns=["image"])
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
        interpoltn = DE_C_INTER_MODE[interpolation]
        super().__init__(*size, *scale, *ratio, interpoltn, max_attempts)


class CenterCrop(cde.CenterCropOp):
    """
    Crops the input image at the center to the given size.

    Args:
        size (Union[int, sequence]): The output size of the cropped image.
            If size is an integer, a square crop of size (size, size) is returned.
            If size is a sequence of length 2, it should be (height, width).

    Examples:
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>>
        >>> # crop image to a square
        >>> transforms_list1 = [c_vision.Decode(), c_vision.CenterCrop(50)]
        >>> data1 = data1.map(operations=transforms_list1, input_columns=["image"])
        >>> # crop image to portrait style
        >>> transforms_list2 = [c_vision.Decode(), c_vision.CenterCrop((60, 40))]
        >>> data2 = data2.map(operations=transforms_list2, input_columns=["image"])
    """

    @check_crop
    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(*size)


class RandomColor(cde.RandomColorOp):
    """
    Adjust the color of the input image by a fixed or random degree.
    This operation works only with 3-channel color images.

    Args:
         degrees (sequence, optional): Range of random color adjustment degrees.
            It should be in (min, max) format. If min=max, then it is a
            single fixed magnitude operation (default=(0.1, 1.9)).

    Examples:
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>>
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomColor((0.5, 2.0))]
        >>> data1 = data1.map(operations=transforms_list, input_columns=["image"])
    """

    @check_positive_degrees
    def __init__(self, degrees=(0.1, 1.9)):
        super().__init__(*degrees)


class RandomColorAdjust(cde.RandomColorAdjustOp):
    """
    Randomly adjust the brightness, contrast, saturation, and hue of the input image.

    Args:
        brightness (Union[float, tuple], optional): Brightness adjustment factor (default=(1, 1)). Cannot be negative.
            If it is a float, the factor is uniformly chosen from the range [max(0, 1-brightness), 1+brightness].
            If it is a sequence, it should be [min, max] for the range.
        contrast (Union[float, tuple], optional): Contrast adjustment factor (default=(1, 1)). Cannot be negative.
            If it is a float, the factor is uniformly chosen from the range [max(0, 1-contrast), 1+contrast].
            If it is a sequence, it should be [min, max] for the range.
        saturation (Union[float, tuple], optional): Saturation adjustment factor (default=(1, 1)). Cannot be negative.
            If it is a float, the factor is uniformly chosen from the range [max(0, 1-saturation), 1+saturation].
            If it is a sequence, it should be [min, max] for the range.
        hue (Union[float, tuple], optional): Hue adjustment factor (default=(0, 0)).
            If it is a float, the range will be [-hue, hue]. Value should be 0 <= hue <= 0.5.
            If it is a sequence, it should be [min, max] where -0.5 <= min <= max <= 0.5.

    Examples:
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>>
        >>> decode_op = c_vision.Decode()
        >>> transform_op = c_vision.RandomColorAdjust(brightness=(0.5, 1), contrast=(0.4, 1), saturation=(0.3, 1))
        >>> transforms_list = [decode_op, transform_op]
        >>> data1 = data1.map(operations=transforms_list, input_columns=["image"])
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

        super().__init__(*brightness, *contrast, *saturation, *hue)

    def expand_values(self, value, center=1, bound=(0, FLOAT_MAX_INTEGER), non_negative=True):
        if isinstance(value, numbers.Number):
            value = [center - value, center + value]
            if non_negative:
                value[0] = max(0, value[0])
            check_range(value, bound)
        return (value[0], value[1])


class RandomRotation(cde.RandomRotationOp):
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
        fill_value (Union[int, tuple], optional): Optional fill color for the area outside the rotated image
            (default=0).
            If it is a 3-tuple, it is used for R, G, B channels respectively.
            If it is an integer, it is used for all RGB channels.

    Examples:
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>> from mindspore.dataset.transforms.vision import Inter
        >>>
        >>> transforms_list = [c_vision.Decode(),
        >>>                    c_vision.RandomRotation(degrees=5.0, resample=Inter.NEAREST, expand=True)]
        >>> data1 = data1.map(operations=transforms_list, input_columns=["image"])
    """

    @check_random_rotation
    def __init__(self, degrees, resample=Inter.NEAREST, expand=False, center=None, fill_value=0):
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill_value = fill_value
        if isinstance(degrees, numbers.Number):
            degrees = (-degrees, degrees)
        if center is None:
            center = (-1, -1)
        if isinstance(fill_value, int):
            fill_value = tuple([fill_value] * 3)
        interpolation = DE_C_INTER_MODE[resample]
        super().__init__(*degrees, *center, interpolation, expand, *fill_value)


class Rescale(cde.RescaleOp):
    """
    Tensor operation to rescale the input image.

    Args:
        rescale (float): Rescale factor.
        shift (float): Shift factor.

    Examples:
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>>
        >>> transforms_list = [c_vision.Decode(), c_vision.Rescale(1.0 / 255.0, -1.0)]
        >>> data1 = data1.map(operations=transforms_list, input_columns=["image"])
    """

    @check_rescale
    def __init__(self, rescale, shift):
        self.rescale = rescale
        self.shift = shift
        super().__init__(rescale, shift)

    def __call__(self, img):
        """
        Call method.

        Args:
            img (NumPy or PIL image): Image to be rescaled.

        Returns:
            img (NumPy), Rescaled image.
        """
        if not isinstance(img, (np.ndarray, Image.Image)):
            raise TypeError("Input should be NumPy or PIL image, got {}.".format(type(img)))
        rescale = cde.Execute(cde.RescaleOp(self.rescale, self.shift))
        img = rescale(cde.Tensor(np.asarray(img)))
        return img.as_array()


class RandomResize(cde.RandomResizeOp):
    """
    Tensor operation to resize the input image using a randomly selected interpolation mode.

    Args:
        size (Union[int, sequence]): The output size of the resized image.
            If size is an integer, smaller edge of the image will be resized to this value with
            the same image aspect ratio.
            If size is a sequence of length 2, it should be (height, width).

    Examples:
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>>
        >>> # randomly resize image, keeping aspect ratio
        >>> transforms_list1 = [c_vision.Decode(), c_vision.RandomResize(50)]
        >>> data1 = data1.map(operations=transforms_list1, input_columns=["image"])
        >>> # randomly resize image to landscape style
        >>> transforms_list2 = [c_vision.Decode(), c_vision.RandomResize((40, 60))]
        >>> data2 = data2.map(operations=transforms_list2, input_columns=["image"])
    """

    @check_resize
    def __init__(self, size):
        self.size = size
        if isinstance(size, int):
            size = (size, 0)
        super().__init__(*size)


class RandomResizeWithBBox(cde.RandomResizeWithBBoxOp):
    """
    Tensor operation to resize the input image using a randomly selected interpolation mode and adjust
    bounding boxes accordingly.

    Args:
        size (Union[int, sequence]): The output size of the resized image.
            If size is an integer, smaller edge of the image will be resized to this value with
            the same image aspect ratio.
            If size is a sequence of length 2, it should be (height, width).

    Examples:
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>>
        >>> # randomly resize image with bounding boxes, keeping aspect ratio
        >>> transforms_list1 = [c_vision.Decode(), c_vision.RandomResizeWithBBox(60)]
        >>> data1 = data1.map(operations=transforms_list1, input_columns=["image"])
        >>> # randomly resize image with bounding boxes to portrait style
        >>> transforms_list2 = [c_vision.Decode(), c_vision.RandomResizeWithBBox((80, 60))]
        >>> data2 = data2.map(operations=transforms_list2, input_columns=["image"])
    """

    @check_resize
    def __init__(self, size):
        self.size = size
        if isinstance(size, int):
            size = (size, 0)
        super().__init__(*size)


class HWC2CHW(cde.ChannelSwapOp):
    """
    Transpose the input image; shape (H, W, C) to shape (C, H, W).

    Examples:
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>>
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomHorizontalFlip(0.75), c_vision.RandomCrop(512),
        >>>     c_vision.HWC2CHW()]
        >>> data1 = data1.map(operations=transforms_list, input_columns=["image"])
    """

    def __call__(self, img):
        """
        Call method.

        Args:
            img (NumPy or PIL image): Image array, of shape (H, W, C), to have channels swapped.

        Returns:
            img (NumPy), Image array, of shape (C, H, W), with channels swapped.
        """
        if not isinstance(img, (np.ndarray, Image.Image)):
            raise TypeError("Input should be NumPy or PIL image, got {}.".format(type(img)))
        hwc2chw = cde.Execute(cde.ChannelSwapOp())
        img = hwc2chw(cde.Tensor(np.asarray(img)))
        return img.as_array()


class RandomCropDecodeResize(cde.RandomCropDecodeResizeOp):
    """
    Equivalent to RandomResizedCrop, but crops before decodes.

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
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>> from mindspore.dataset.transforms.vision import Inter
        >>>
        >>> resize_crop_decode_op = c_vision.RandomCropDecodeResize(size=(50, 75), scale=(0.25, 0.5),
        >>>     interpolation=Inter.NEAREST, max_attempts=5)
        >>> transforms_list = [resize_crop_decode_op]
        >>> data1 = data1.map(operations=transforms_list, input_columns=["image"])
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
        interpoltn = DE_C_INTER_MODE[interpolation]
        super().__init__(*size, *scale, *ratio, interpoltn, max_attempts)


class Pad(cde.PadOp):
    """
    Pads the image according to padding parameters.

    Args:
        padding (Union[int, sequence]): The number of pixels to pad the image.
            If a single number is provided, it pads all borders with this value.
            If a tuple or list of 2 values are provided, it pads the (left and top)
            with the first value and (right and bottom) with the second value.
            If 4 values are provided as a list or tuple,
            it pads the left, top, right and bottom respectively.
        fill_value (Union[int, tuple], optional): The pixel intensity of the borders, only valid for
            padding_mode Border.CONSTANT (default=0).
            If it is an integer, it is used for all RGB channels.
            If it is a 3-tuple, it is used to fill R, G, B channels respectively.
            The fill_value values must be in range [0, 255].
        padding_mode (Border mode, optional): The method of padding (default=Border.CONSTANT). Can be any of
            [Border.CONSTANT, Border.EDGE, Border.REFLECT, Border.SYMMETRIC].

            - Border.CONSTANT, means it fills the border with constant values.

            - Border.EDGE, means it pads with the last value on the edge.

            - Border.REFLECT, means it reflects the values on the edge omitting the last
              value of edge.

            - Border.SYMMETRIC, means it reflects the values on the edge repeating the last
              value of edge.

    Examples:
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>> from mindspore.dataset.transforms.vision import Border
        >>>
        >>> transforms_list = [c_vision.Decode(), c_vision.Pad([100, 100, 100, 100])]
        >>> data1 = data1.map(operations=transforms_list, input_columns=["image"])
    """

    @check_pad
    def __init__(self, padding, fill_value=0, padding_mode=Border.CONSTANT):
        padding = parse_padding(padding)
        if isinstance(fill_value, int):
            fill_value = tuple([fill_value] * 3)
        self.padding = padding
        self.fill_value = fill_value
        self.padding_mode = padding_mode
        padding_mode = DE_C_BORDER_TYPE[padding_mode]
        super().__init__(*padding, padding_mode, *fill_value)

    def __call__(self, img):
        """
        Call method.

        Args:
            img (NumPy or PIL image): Image to be padded.

        Returns:
            img (NumPy), Padded image.
        """
        if not isinstance(img, (np.ndarray, Image.Image)):
            raise TypeError("Input should be NumPy or PIL image, got {}.".format(type(img)))
        pad = cde.Execute(cde.PadOp(*self.padding, DE_C_BORDER_TYPE[self.padding_mode], *self.fill_value))
        img = pad(cde.Tensor(np.asarray(img)))
        return img.as_array()


class UniformAugment(cde.UniformAugOp):
    """
    Tensor operation to perform randomly selected augmentation.

    Args:
        transforms: List of C++ operations (Python operations are not accepted).
        num_ops (int, optional): Number of operations to be selected and applied (default=2).

    Examples:
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>> import mindspore.dataset.vision.py_transforms as py_vision
        >>>
        >>> transforms_list = [c_vision.RandomHorizontalFlip(),
        >>>                    c_vision.RandomVerticalFlip(),
        >>>                    c_vision.RandomColorAdjust(),
        >>>                    c_vision.RandomRotation(degrees=45)]
        >>> uni_aug_op = c_vision.UniformAugment(transforms=transforms_list, num_ops=2)
        >>> transforms_all = [c_vision.Decode(), c_vision.Resize(size=[224, 224]),
        >>>                   uni_aug_op, py_vision.ToTensor()]
        >>> data_aug = data1.map(operations=transforms_all, input_columns="image",
        >>>                      num_parallel_workers=1)
    """

    @check_uniform_augment_cpp
    def __init__(self, transforms, num_ops=2):
        self.transforms = transforms
        self.num_ops = num_ops
        super().__init__(transforms, num_ops)


class RandomSelectSubpolicy(cde.RandomSelectSubpolicyOp):
    """
    Choose a random sub-policy from a list to be applied on the input image. A sub-policy is a list of tuples
    (op, prob), where op is a TensorOp operation and prob is the probability that this op will be applied. Once
    a sub-policy is selected, each op within the subpolicy with be applied in sequence according to its probability.

    Args:
        policy (list(list(tuple(TensorOp,float))): List of sub-policies to choose from.

    Examples:
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>>
        >>> policy = [[(c_vision.RandomRotation((45, 45)), 0.5), (c_vision.RandomVerticalFlip(), 1),
        >>>            (c_vision.RandomColorAdjust(), 0.8)],
        >>>           [(c_vision.RandomRotation((90, 90)), 1), (c_vision.RandomColorAdjust(), 0.2)]]
        >>> data_policy = data1.map(operations=c_vision.RandomSelectSubpolicy(policy), input_columns=["image"])
    """

    @check_random_select_subpolicy_op
    def __init__(self, policy):
        super().__init__(policy)


class SoftDvppDecodeResizeJpeg(cde.SoftDvppDecodeResizeJpegOp):
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
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>>
        >>> # decode and resize image, keeping aspect ratio
        >>> transforms_list1 = [c_vision.Decode(), c_vision.SoftDvppDecodeResizeJpeg(70)]
        >>> data1 = data1.map(operations=transforms_list1, input_columns=["image"])
        >>> # decode and resize to portrait style
        >>> transforms_list2 = [c_vision.Decode(), c_vision.SoftDvppDecodeResizeJpeg((80, 60))]
        >>> data2 = data2.map(operations=transforms_list2, input_columns=["image"])
    """

    @check_resize
    def __init__(self, size):
        if isinstance(size, int):
            size = (size, 0)
        self.size = size
        super().__init__(*size)


class SoftDvppDecodeRandomCropResizeJpeg(cde.SoftDvppDecodeRandomCropResizeJpegOp):
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
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>>
        >>> # decode, randomly crop and resize image, keeping aspect ratio
        >>> transforms_list1 = [c_vision.Decode(), c_vision.SoftDvppDecodeRandomCropResizeJpeg(90)]
        >>> data1 = data1.map(operations=transforms_list1, input_columns=["image"])
        >>> # decode, randomly crop and resize to landscape style
        >>> transforms_list2 = [c_vision.Decode(), c_vision.SoftDvppDecodeRandomCropResizeJpeg((80, 100))]
        >>> data2 = data2.map(operations=transforms_list2, input_columns=["image"])
    """

    @check_soft_dvpp_decode_random_crop_resize_jpeg
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), max_attempts=10):
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.max_attempts = max_attempts
        super().__init__(*size, *scale, *ratio, max_attempts)


class RandomSolarize(cde.RandomSolarizeOp):
    """
    Invert all pixel values above a threshold.

    Args:
        threshold (tuple, optional): Range of random solarize threshold. Threshold values should always be
            in the range (0, 255), include at least one integer value in the given range and
            be in (min, max) format. If min=max, then it is a single fixed magnitude operation (default=(0, 255)).

    Examples:
        >>> import mindspore.dataset.vision.c_transforms as c_vision
        >>>
        >>> transforms_list = [c_vision.Decode(), c_vision.RandomSolarize(threshold=(10,100))]
        >>> data1 = data1.map(operations=transforms_list, input_columns=["image"])
    """

    @check_random_solarize
    def __init__(self, threshold=(0, 255)):
        super().__init__(threshold)
