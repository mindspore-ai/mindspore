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
from .._primitive_cache import _get_cache_prim


def bounding_box_decode(anchor_box, deltas, max_shape, means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0),
                        wh_ratio_clip=0.016):
    r"""
    Decodes bounding boxes locations.

    The function of the operator is to calculate the offset, and this operator converts the offset into a Bbox,
    which is used to mark the target in the subsequent images, etc.

    Args:
        anchor_box (Tensor): Anchor boxes. The shape of `anchor_box` must be :math:`(n, 4)`.
        deltas (Tensor): Delta of boxes. Which has the same shape with `anchor_box`.
        max_shape (tuple): The max size limit for decoding box calculation.
        means (tuple): The means of `deltas` calculation. Default: (0.0, 0.0, 0.0, 0.0).
        stds (tuple): The standard deviations of `deltas` calculation. Default: (1.0, 1.0, 1.0, 1.0).
        wh_ratio_clip (float): The limit of width and height ratio for decoding box calculation. Default: 0.016.

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
    Encodes bounding boxes locations.

    This operator will calculate the offset between the predicted bounding boxes and the real bounding boxes,
    and this offset will be used as a variable for the loss.

    Args:
        anchor_box (Tensor): Anchor boxes. The shape of `anchor_box` must be :math:`(n, 4)`.
        groundtruth_box (Tensor): Ground truth boxes. Which has the same shape with `anchor_box`.
        means (tuple): Means for encoding bounding boxes calculation. Default: (0.0, 0.0, 0.0, 0.0).
        stds (tuple): The standard deviations of deltas calculation. Default: (1.0, 1.0, 1.0, 1.0).

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
        >>> output = ops.bounding_box_encode(anchor_box, groundtruth_box, means=(0.0, 0.0, 0.0, 0.0)
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
        specifying the valid boundary (heights x ratio, weights x ratio).

    Args:
        bboxes (Tensor): Bounding boxes tensor with shape :math:`(N, 4)`. :math:`N` indicates the number of
            bounding boxes, the value `4` indicates `x0`, `x1`, `y0, and `y1`. Data type must be float16 or float32.
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


__all__ = [
    'bounding_box_decode',
    'bounding_box_encode',
    'check_valid'
]

__all__.sort()
