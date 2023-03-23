# Copyright 2022 Huawei Technologies Co., Ltd
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
# ============================================================================

"""Defines image operators with functional form."""

from mindspore.ops import operations as P
from mindspore.ops.operations import image_ops as IMG
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore._c_expression import Tensor as Tensor_
from .._primitive_cache import _get_cache_prim


def bounding_box_decode(anchor_box, deltas, max_shape, means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0),
                        wh_ratio_clip=0.016):
    r"""
    Decode the bounding box locations, calculate the offset, and convert the offset into a Bbox, which is used to mark
    the target in the subsequent images, etc.

    Args:
        anchor_box (Tensor): Anchor boxes. The shape of `anchor_box` must be :math:`(n, 4)`.
        deltas (Tensor): Delta of boxes. Which has the same shape with `anchor_box`.
        max_shape (tuple): The max size limit for decoding box calculation.
        means (tuple, optional): The means of `deltas` calculation. Default: (0.0, 0.0, 0.0, 0.0).
        stds (tuple, optional): The standard deviations of `deltas` calculation. Default: (1.0, 1.0, 1.0, 1.0).
        wh_ratio_clip (float, optional): The limit of width and height ratio for decoding box calculation.
            Default: 0.016.

    Returns:
        Tensor, decoded boxes. It has the same data type and shape as `anchor_box`.

    Raises:
        TypeError: If `means`, `stds` or `max_shape` is not a tuple.
        TypeError: If `wh_ratio_clip` is not a float.
        TypeError: If `anchor_box` or `deltas` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> anchor_box = Tensor([[4, 1, 2, 1], [2, 2, 2, 3]], mindspore.float32)
        >>> deltas = Tensor([[3, 1, 2, 2], [1, 2, 1, 4]], mindspore.float32)
        >>> output = ops.bounding_box_decode(anchor_box, deltas, max_shape=(768, 1280), means=(0.0, 0.0, 0.0, 0.0),
        ...                                  stds=(1.0, 1.0, 1.0, 1.0), wh_ratio_clip=0.016)
        >>> print(output)
        [[ 4.1953125  0.         0.         5.1953125]
         [ 2.140625   0.         3.859375  60.59375  ]]
    """

    bounding_box_decode_op = _get_cache_prim(P.BoundingBoxDecode)(max_shape, means, stds, wh_ratio_clip)
    return bounding_box_decode_op(anchor_box, deltas)


def bounding_box_encode(anchor_box, groundtruth_box, means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0)):
    r"""
    Encode the bounding box locations, calculate the offset between the predicted bounding boxes and
    the real bounding boxes, and the offset will be used as a variable for the loss.

    Args:
        anchor_box (Tensor): Anchor boxes. The shape of `anchor_box` must be :math:`(n, 4)`.
        groundtruth_box (Tensor): Ground truth boxes. Which has the same shape with `anchor_box`.
        means (tuple, optional): Means for encoding bounding boxes calculation. Default: (0.0, 0.0, 0.0, 0.0).
        stds (tuple, optional): The standard deviations of deltas calculation. Default: (1.0, 1.0, 1.0, 1.0).

    Returns:
        Tensor, encoded bounding boxes. It has the same data type and shape as input `anchor_box`.

    Raises:
        TypeError: If `means` or `stds` is not a tuple.
        TypeError: If `anchor_box` or `groundtruth_box` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> anchor_box = Tensor([[2, 2, 2, 3], [2, 2, 2, 3]], mindspore.float32)
        >>> groundtruth_box = Tensor([[1, 2, 1, 4], [1, 2, 1, 4]], mindspore.float32)
        >>> output = ops.bounding_box_encode(anchor_box, groundtruth_box, means=(0.0, 0.0, 0.0, 0.0),
        ...                                  stds=(1.0, 1.0, 1.0, 1.0))
        >>> print(output)
        [[ -1.  0.25  0.  0.40551758]
         [ -1.  0.25  0.  0.40551758]]
    """

    bounding_box_encode_op = _get_cache_prim(P.BoundingBoxEncode)(means, stds)
    return bounding_box_encode_op(anchor_box, groundtruth_box)


def check_valid(bboxes, img_metas):
    r"""
    Checks bounding box.

    Checks whether the bounding box cross data and data border are valid.

    .. warning::
        Boundary(heights * ratio, widths * ratio) specified by `bboxes` is required to be valid.

    Args:
        bboxes (Tensor): Bounding boxes tensor with shape :math:`(N, 4)`. :math:`N` indicates the number of
            bounding boxes, the value `4` indicates `x0`, `x1`, `y0`, and `y1`. Data type must be float16 or float32.
        img_metas (Tensor): Raw image size information with the format of `(height, width, ratio)`, specifying
            the valid boundary `(height * ratio, width * ratio)`. Data type must be float16 or float32.

    Returns:
        Tensor, with shape of :math:`(N,)` and dtype of bool, specifying whether the bounding boxes is in the image.
        `True` indicates valid, while `False` indicates invalid.

    Raises:
        TypeError: If `bboxes` or `img_metas` is not a Tensor.
        TypeError: If dtype of `bboxes` or `img_metas` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> bboxes = Tensor(np.linspace(0, 6, 12).reshape(3, 4), mindspore.float32)
        >>> img_metas = Tensor(np.array([2, 1, 3]), mindspore.float32)
        >>> output = ops.check_valid(bboxes, img_metas)
        >>> print(output)
        [ True False False]
    """
    check_valid_op = _get_cache_prim(P.CheckValid)()
    return check_valid_op(bboxes, img_metas)


def crop_and_resize(image, boxes, box_indices, crop_size, method="bilinear", extrapolation_value=0.0):
    """
    Extracts crops from the input image Tensor and resizes them.

    Note:
        In case that the output shape depends on crop_size, the crop_size must be constant.
        For now, the backward of the operator only support bilinear method, for other methods, will return 0.

    Args:
        image (Tensor): The input image must be a 4-D tensor of shape (batch, image_height, image_width, depth).
           Types allowed: int8, int16, int32, int64, float16, float32, float64, uint8, uint16.
        boxes (Tensor): A 2-D tensor of shape [num_boxes, 4].
           The i-th row of the tensor specifies the coordinates of a box in the box_ind[i] image
           and is specified in normalized coordinates [y1, x1, y2, x2]. A normalized coordinate value of y is mapped to
           the image coordinate at y * (image_height - 1), so as the [0, 1] interval of normalized image height is
           mapped to [0, image_height - 1] in image height coordinates. We do allow y1 > y2, in which case the sampled
           crop is an up-down flipped version of the original image. The width dimension is treated similarly.
           Normalized coordinates outside the [0, 1] range are allowed, in which case we use extrapolation_value to
           extrapolate the input image values. Types allowed: float32.
        box_indices (Tensor): A 1-D tensor of shape [num_boxes] with int32 values in [0, batch).
           The value of box_ind[i] specifies the image that the i-th box refers to. Types allowed: int32.
        crop_size (Tuple[int]): A tuple of two int32 elements: (crop_height, crop_width).
           Only constant value is allowed. All cropped image patches are resized to this size.
           The aspect ratio of the image content is not preserved. Both crop_height and crop_width need to be positive.
        method (str, optional): An optional string that specifies the sampling method for resizing.
           It can be "bilinear", "nearest" or "bilinear_v2". The option "bilinear" stands for standard bilinear
           interpolation algorithm, while "bilinear_v2" may result in better result in some cases. Default: "bilinear"
        extrapolation_value (float, optional): An optional float value used extrapolation, if applicable. Default: 0.0.

    Returns:
        A 4-D tensor of shape [num_boxes, crop_height, crop_width, depth] with type(float32).

    Raises:
        TypeError: If `image` or `boxes` or `box_indices` is not a Tensor.
        TypeError: If `crop_size` is not a Tuple with two int32 elements.
        TypeError: If dtype of `boxes` is not float or that of `box_indices` is not int.
        TypeError: If `method` is not a str.
        TypeError: If `extrapolation_value` is not a float.
        ValueError: If the shape rank of `image` is not 4.
        ValueError: If the shape rank of `boxes` is not 2.
        ValueError: If the second dim of `boxes` is not 4.
        ValueError: If the shape rank of `box_indices` is not 1.
        ValueError: If the first dim of `box_indices` is not equal to that of `boxes`.
        ValueError: If existing element in `box_indices` is out of range `[0, batch)`.
        ValueError: If the data of `crop_size` is not positive.
        ValueError: If `method` is not one of 'bilinear', 'nearest', 'bilinear_v2'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> BATCH_SIZE = 1
        >>> NUM_BOXES = 5
        >>> IMAGE_HEIGHT = 256
        >>> IMAGE_WIDTH = 256
        >>> CHANNELS = 3
        >>> image = np.random.normal(size=[BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS]).astype(np.float32)
        >>> boxes = np.random.uniform(size=[NUM_BOXES, 4]).astype(np.float32)
        >>> box_indices = np.random.uniform(size=[NUM_BOXES], low=0, high=BATCH_SIZE).astype(np.int32)
        >>> crop_size = (24, 24)
        >>> output = ops.crop_and_resize(Tensor(image), Tensor(boxes), Tensor(box_indices), crop_size)
        >>> print(output.shape)
         (5, 24, 24, 3)
    """
    if not isinstance(image, (Tensor, Tensor_)):
        raise TypeError("For crop_and_resize, the input image must be a tensor")
    if not isinstance(boxes, (Tensor, Tensor_)):
        raise TypeError("For crop_and_resize, the input boxes must be a tensor")
    if not isinstance(box_indices, (Tensor, Tensor_)):
        raise TypeError("For crop_and_resize, the input box_indices must be a tensor")
    if not isinstance(crop_size, tuple):
        raise TypeError("For crop_and_resize, the input crop_size must be a tuple, but got {}".format(type(crop_size)))
    if len(crop_size) != 2:
        raise ValueError("For crop_and_resize, the crop_size's length must be 2, bot got {}".format(len(crop_size)))
    if not isinstance(crop_size[0], int) or not isinstance(crop_size[1], int):
        raise TypeError("For crop_and_resize, the crop_size's value must be int.")
    if crop_size[0] <= 0 or crop_size[1] <= 0:
        raise ValueError("For crop_and_resize, the crop_size's value must be positive.")

    image_shape = image.shape
    if len(image_shape) != 4:
        raise ValueError(
            "For crop_and_resize, the input image must be 4D Tensor, but got is {}D".format(len(image_shape)))
    boxes_dtype = _get_cache_prim(P.DType)()(boxes)
    if boxes_dtype not in [mstype.float32]:
        raise TypeError(
            "For crop_and_resize, the input boxes must be {}, but got {}".format(mstype.float32, boxes_dtype))
    boxes_shape = boxes.shape
    if len(boxes_shape) != 2 or boxes_shape[-1] != 4:
        raise ValueError("For crop_and_resize, the input boxes must be 2D and the second-dim must be 4, "
                         "but got {}".format(boxes_shape))
    box_indices_dtype = _get_cache_prim(P.DType)()(box_indices)
    if box_indices_dtype not in [mstype.int32]:
        raise TypeError(
            "For crop_and_resize, the input box_indices must be {}, but got {}".format(mstype.int32, box_indices_dtype))
    box_indices_shape = box_indices.shape
    if len(box_indices_shape) != 1:
        raise ValueError("For crop_and_resize, the input box_indices must be 1D, but got {}".format(box_indices_shape))
    if boxes_shape[0] != box_indices_shape[0]:
        raise ValueError("For crop_and_resize, the first dim of input box_indices must be equal to that of input boxes"
                         ", but got {} vs {}".format(box_indices_shape[0], boxes_shape[0]))

    _crop_and_resize = _get_cache_prim(IMG.CropAndResize)(method, extrapolation_value)
    return _crop_and_resize(image, boxes, box_indices, crop_size)


__all__ = [
    'bounding_box_decode',
    'bounding_box_encode',
    'check_valid',
    'crop_and_resize'
]

__all__.sort()
