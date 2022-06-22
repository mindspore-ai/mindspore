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

from ..operations import image_ops as IMG


def resize_bilinear(x, size, align_corners=False, half_pixel_centers=False):
    r"""
    Resizes an image to a certain size using the bilinear interpolation.

    The resizing only affects the lower two dimensions which represent the height and width.

    Args:
        x (Tensor): Image to be resized. Input images must be a 4-D tensor with shape
            :math:`(batch, channels, height, width)`, with data type of float32 or float16.
        size (Union[tuple[int], list[int], Tensor]): The new size of the images.
            A tuple or list or Tensor of 2 int elements :math:`(new\_height, new\_width)`.
        align_corners (bool): If true, rescale input by :math:`(new\_height - 1) / (height - 1)`,
                       which exactly aligns the 4 corners of images and resized images. If false,
                       rescale by :math:`new\_height / height`. Default: False.
        half_pixel_centers (bool): Whether half pixel center. If set to True, `align_corners` should be False.
                           Default: False.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Returns:
        Tensor, resized image. 4-D with shape :math:`(batch, channels, new\_height, new\_width)`,
        with the same data type as input `x`.

    Raises:
        TypeError: If `align_corners` is not a bool.
        TypeError: If `half_pixel_centers` is not a bool.
        TypeError: If `align_corners` and `half_pixel_centers` are all True.
        ValueError: If `half_pixel_centers` is True and device_target is CPU.

    Supported Platforms:
        ``Ascend`` ``CPU`` ``GPU``

    Examples:
        >>> x = Tensor([[[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]]], mindspore.float32)
        >>> output = resize_bilinear(x, (5, 5))
        >>> print(output)
        [[[[1. 2. 3. 4. 5.]
           [1. 2. 3. 4. 5.]
           [1. 2. 3. 4. 5.]
           [1. 2. 3. 4. 5.]
           [1. 2. 3. 4. 5.]]]]
    """
    return IMG.ResizeBilinearV2(align_corners, half_pixel_centers)(x, size)


__all__ = [
    'resize_bilinear'
]
