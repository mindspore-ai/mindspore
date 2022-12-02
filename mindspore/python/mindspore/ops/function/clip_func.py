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

"""Defines clip operators with functional form."""

from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.common.tensor import Tensor

__all__ = [
    'clip_by_value',
    'clamp',
    'clip',
]

hyper_map = C.HyperMap()
max_op = _get_cache_prim(P.Maximum)()
min_op = _get_cache_prim(P.Minimum)()
cast_op = _get_cache_prim(P.Cast)()
scalar2tensor_op = _get_cache_prim(P.ScalarToTensor)()
partial_op = _get_cache_prim(P.Partial)()


def clip_by_value(x, clip_value_min=None, clip_value_max=None):
    r"""
    Clips tensor values to a specified min and max.

    Limits the value of :math:`x` to a range, whose lower limit is `clip_value_min`
    and upper limit is `clip_value_max` .

    .. math::

        out_i= \left\{
        \begin{array}{align}
            clip\_value\_max & \text{ if } x_i\ge  clip\_value\_max \\
            x_i & \text{ if } clip\_value\_min \lt x_i \lt clip\_value\_max \\
            clip\_value\_min & \text{ if } x_i \le clip\_value\_min \\
        \end{array}\right.

    Note:
        - `clip_value_min` and `clip_value_max` cannot be None at the same time;
        - When `clip_value_min` is None and `clip_value_max` is not None, the elements in Tensor
          larger than `clip_value_max` will become `clip_value_max`;
        - When `clip_value_min` is not None and `clip_value_max` is None, the elements in Tensor
          smaller than `clip_value_min` will become `clip_value_min`;
        - If `clip_value_min` is greater than `clip_value_max`, the value of all elements in Tensor
          will be set to `clip_value_max`;
        - The data type of `x`, `clip_value_min` and `clip_value_max` should support implicit type
          conversion and cannot be bool type.

    Args:
          x (Union(Tensor, list[Tensor], tuple[Tensor])): Input data, which type is Tensor or a list or tuple of Tensor.
                                                         The shape of Tensor is :math:`(N,*)` where :math:`*` means,
                                                         any number of additional dimensions.
          clip_value_min (Union(Tensor, float, int)): The minimum value. Default: None.
          clip_value_max (Union(Tensor, float, int)): The maximum value. Default: None.

    Returns:
          (Union(Tensor, tuple[Tensor], list[Tensor])), a clipped Tensor or a tuple or a list of clipped Tensor.
          The data type and shape are the same as x.

    Raises:
          ValueError: If both `clip_value_min` and `clip_value_max` are None.
          TypeError: If the type of `x` is not in Tensor or list[Tensor] or tuple[Tensor].
          TypeError: If the type of `clip_value_min` is not in None, Tensor, float or int.
          TypeError: If the type of `clip_value_max` is not in None, Tensor, float or int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # case 1: the data type of x is Tensor
        >>> import mindspore
        >>> from mindspore import Tensor, ops
        >>> import numpy as np
        >>> min_value = Tensor(5, mindspore.float32)
        >>> max_value = Tensor(20, mindspore.float32)
        >>> x = Tensor(np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]), mindspore.float32)
        >>> output = ops.clip_by_value(x, min_value, max_value)
        >>> print(output)
        [[ 5. 20.  5.  7.]
         [ 5. 11.  6. 20.]]
        >>> # case 2: the data type of x is list[Tensor]
        >>> min_value = 5
        >>> max_value = 20
        >>> x = Tensor(np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]), mindspore.float32)
        >>> y = Tensor(np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]), mindspore.float32)
        >>> output = ops.clip_by_value([x,y], min_value, max_value)
        >>> print(output)
        [[[ 5. 20.  5.  7.]
          [ 5. 11.  6. 20.]],
         [[ 5. 20.  5.  7.]
          [ 5. 11.  6. 20.]]]
    """
    def _clip_by_value(clip_min, clip_max, x):
        if not isinstance(x, Tensor):
            raise TypeError("Then type of 'x' must be Tensor")
        result = x
        if clip_min is not None:
            result = max_op(result, cast_op(clip_min, x.dtype))
        if clip_max is not None:
            result = min_op(result, cast_op(clip_max, x.dtype))
        return result

    if clip_value_min is None and clip_value_max is None:
        raise ValueError("At least one of 'clip_value_min' or 'clip_value_max' must not be None")
    if not isinstance(x, (Tensor, tuple, list)):
        raise TypeError("The input of 'clip_by_value' must be tensor or tuple[Tensor] or list[Tensor]")
    if not isinstance(clip_value_min, (type(None), Tensor, float, int)):
        raise TypeError("Then type of 'clip_value_min' must be not one of None, Tensor, float, int.")
    if not isinstance(clip_value_max, (type(None), Tensor, float, int)):
        raise TypeError("Then type of 'clip_value_max' must be not one of None, Tensor, float, int.")
    if isinstance(clip_value_min, (float, int)):
        clip_value_min = scalar2tensor_op(clip_value_min)
    if isinstance(clip_value_max, (float, int)):
        clip_value_max = scalar2tensor_op(clip_value_max)

    if isinstance(x, Tensor):
        return _clip_by_value(clip_value_min, clip_value_max, x)
    results = hyper_map(partial_op(_clip_by_value, clip_value_min, clip_value_max), x)
    if isinstance(x, tuple):
        results = tuple(results)
    return results


def clamp(x, min=None, max=None):
    r"""
    Clamps tensor values to a specified min and max.

    Limits the value of :math:`x` to a range, whose lower limit is `min` and upper limit is `max` .

    .. math::

        out_i= \left\{
        \begin{array}{align}
            max & \text{ if } x_i\ge  max \\
            x_i & \text{ if } min \lt x_i \lt max \\
            min & \text{ if } x_i \le min \\
        \end{array}\right.

    Note:
        - `min` and `max` cannot be None at the same time;
        - When `min` is None and `max` is not None, the elements in Tensor larger than `max` will become `max`;
        - When `min` is not None and `max` is None, the elements in Tensor smaller than `min` will become `min`;
        - If `min` is greater than `max`, the value of all elements in Tensor will be set to `max`;
        - The data type of `x`, `min` and `max` should support implicit type conversion and cannot be bool type.

    Args:
          x (Union(Tensor, list[Tensor], tuple[Tensor])): Input data, which type is Tensor or a list or tuple of Tensor.
                                                         The shape of Tensor is :math:`(N,*)` where :math:`*` means,
                                                         any number of additional dimensions.
          min (Union(Tensor, float, int)): The minimum value. Default: None.
          max (Union(Tensor, float, int)): The maximum value. Default: None.

    Returns:
          (Union(Tensor, tuple[Tensor], list[Tensor])), a clipped Tensor or a tuple or a list of clipped Tensor.
          The data type and shape are the same as x.

    Raises:
          ValueError: If both `min` and `max` are None.
          TypeError: If the type of `x` is not in Tensor or list[Tensor] or tuple[Tensor].
          TypeError: If the type of `min` is not in None, Tensor, float or int.
          TypeError: If the type of `max` is not in None, Tensor, float or int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # case 1: the data type of x is Tensor
        >>> import mindspore
        >>> from mindspore import Tensor, ops
        >>> import numpy as np
        >>> min_value = Tensor(5, mindspore.float32)
        >>> max_value = Tensor(20, mindspore.float32)
        >>> x = Tensor(np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]), mindspore.float32)
        >>> output = ops.clamp(x, min_value, max_value)
        >>> print(output)
        [[ 5. 20.  5.  7.]
         [ 5. 11.  6. 20.]]
        >>> # case 2: the data type of x is list[Tensor]
        >>> min_value = 5
        >>> max_value = 20
        >>> x = Tensor(np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]), mindspore.float32)
        >>> y = Tensor(np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]), mindspore.float32)
        >>> output = ops.clamp([x,y], min_value, max_value)
        >>> print(output)
        [[[ 5. 20.  5.  7.]
          [ 5. 11.  6. 20.]],
         [[ 5. 20.  5.  7.]
          [ 5. 11.  6. 20.]]]
    """
    return clip_by_value(x, min, max)


def clip(x, min=None, max=None):
    r"""
    Alias for ops.clamp.
    For details, please refer to :func:`mindspore.ops.clamp`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return clamp(x, min, max)
