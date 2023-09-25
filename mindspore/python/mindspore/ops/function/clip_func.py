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
from __future__ import absolute_import

from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.cell import Cell
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype
from mindspore.common.api import jit
from mindspore.ops.primitive import _primexpr
from mindspore import _checkparam as Validator

__all__ = [
    'clip_by_value',
    'clip_by_norm',
    'clamp',
    'clip',
    'clip_by_global_norm',
]

hyper_map = C.HyperMap()
max_op = P.Maximum()
min_op = P.Minimum()
cast_op = P.Cast()
scalar2tensor_op = P.ScalarToTensor()
partial_op = P.Partial()
expand_dims = P.ExpandDims().add_prim_attr("grad_scale", True)
get_square_sum = C.MultitypeFuncGraph("get_square_sum")
apply_global_norm = C.MultitypeFuncGraph("apply_global_norm")


def _old_norm(norm_type, x):
    """Add norm function"""
    out = F.pow((F.reduce_sum(F.pow(x, norm_type))), 1. / norm_type).astype(x.dtype)
    return out


@jit
def _cal_total_norm(x, norm_type):
    if norm_type == float('inf'):
        func = lambda data: data.abs().max()
        total_norm = max(hyper_map(func, x))
    else:
        total_norm = _old_norm(norm_type, F.stack(hyper_map(partial_op(_old_norm, norm_type), x)))
    return total_norm


def clip_by_norm(x, max_norm, norm_type=2.0, error_if_nonfinite=False):
    r"""
    Clip norm of a set of input Tensors. This norm is the result of calculating the norm of all elements in the input
    separately, connecting them into a vector, and then calculating the norm.

    Note:
        The interface is suitable for gradient clipping scenarios, and only supports input of type float.

    Args:
          x (Union(Tensor, list[Tensor], tuple[Tensor])): Input that wishes to be clipped.
          max_norm (Union(float, int)): The upper limit of the norm for this group of network parameters.
          norm_type (Union(float, int)): Norm type. Default: ``2.0``.
          error_if_nonfinite (bool): If it is ``True``, an exception is thrown if the total norm from the input
              is nan, inf or -inf. If it is ``False``, no exception will be thrown.Default: ``False`` .

    Returns:
        Tensors, a list or tuple of Tensors, representing clipped Tensors.

    Raises:
          RuntimeError: If the total norm from the `x` is nan, inf or -inf.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> x = Tensor([[0.8748, 0.1425, 0.0076], [0.7721, 0.4084, 0.0552], [4.6376, 0.2914, 2.1120]])
        >>> out = ops.clip_by_norm(x, max_norm=1)
        >>> print(out)
        [[0.16650201 0.02712224 0.00144652]
         [0.14695495 0.07773139 0.0105063 ]
         [0.8826814  0.0554626  0.40198016]]
    """
    is_tensor = False
    if isinstance(x, Tensor):
        x = [x]
        is_tensor = True
    total_norm = _cal_total_norm(x, norm_type)
    if error_if_nonfinite and F.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(f"For clip_by_norm, the total norm of order {norm_type} from input is non-finite.")
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        ret = hyper_map(partial_op(F.mul, clip_coef), x)
    else:
        ret = x
    if is_tensor:
        return ret[0]
    return ret


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
            Tensors of arbitrary dimensions are supported.
          clip_value_min (Union(Tensor, float, int)): The minimum value. Default: ``None`` .
          clip_value_max (Union(Tensor, float, int)): The maximum value. Default: ``None`` .

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
        >>> for out in output:
        ...     print(out)
        [[ 5. 20.  5.  7.]
         [ 5. 11.  6. 20.]]
        [[ 5. 20.  5.  7.]
         [ 5. 11.  6. 20.]]
    """

    def _clip_by_value(clip_min, clip_max, x):
        if not isinstance(x, Tensor):
            raise TypeError("For 'clip_by_value', the type of argument 'x' must be "
                            "Tensor or tuple[Tensor] or list[Tensor]")
        result = x
        if clip_min is not None:
            result = max_op(result, cast_op(clip_min, x.dtype))
        if clip_max is not None:
            result = min_op(result, cast_op(clip_max, x.dtype))
        return result

    if clip_value_min is None and clip_value_max is None:
        raise ValueError("For 'clip_by_value', at least one of "
                         "'clip_value_min' or 'clip_value_max' must not be None")
    if not isinstance(x, (Tensor, tuple, list)):
        raise TypeError("For 'clip_by_value', the type of argument 'x' must be "
                        "Tensor or tuple[Tensor] or list[Tensor]")
    if not isinstance(clip_value_min, (type(None), Tensor, float, int)):
        raise TypeError("For 'clip_by_value', the type of argument 'clip_value_min' must be "
                        "not one of None, Tensor, float, int")
    if not isinstance(clip_value_max, (type(None), Tensor, float, int)):
        raise TypeError("For 'clip_by_value', the type of argument 'clip_value_max' must be "
                        "not one of None, Tensor, float, int")
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


def clamp(input, min=None, max=None):
    r"""
    Clamps tensor values between the specified minimum value and maximum value.

    Limits the value of :math:`input` to a range, whose lower limit is `min` and upper limit is `max` .

    .. math::

        out_i= \left\{
        \begin{array}{align}
            max & \text{ if } x_i\ge max \\
            x_i & \text{ if } min \lt x_i \lt max \\
            min & \text{ if } x_i \le min \\
        \end{array}\right.

    Note:
        - `min` and `max` cannot be None at the same time;
        - When `min` is None and `max` is not None, the elements in Tensor larger than `max` will become `max`;
        - When `min` is not None and `max` is None, the elements in Tensor smaller than `min` will become `min`;
        - If `min` is greater than `max`, the value of all elements in Tensor will be set to `max`;
        - The data type of `input`, `min` and `max` should support implicit type conversion and cannot be bool type.

    Args:
          input (Union(Tensor, list[Tensor], tuple[Tensor])): Input data, which type is Tensor or a list or tuple of
            Tensor. Tensors of arbitrary dimensions are supported.
          min (Union(Tensor, float, int), optional): The minimum value. Default: ``None`` .
          max (Union(Tensor, float, int), optional): The maximum value. Default: ``None`` .

    Returns:
          Union(Tensor, tuple[Tensor], list[Tensor]), a clipped Tensor or a tuple or a list of clipped Tensor.
          The data type and shape are the same as input.

    Raises:
          ValueError: If both `min` and `max` are None.
          TypeError: If the type of `input` is not in Tensor or list[Tensor] or tuple[Tensor].
          TypeError: If the type of `min` is not in None, Tensor, float or int.
          TypeError: If the type of `max` is not in None, Tensor, float or int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # case 1: the data type of input is Tensor
        >>> import mindspore
        >>> from mindspore import Tensor, ops
        >>> import numpy as np
        >>> min_value = Tensor(5, mindspore.float32)
        >>> max_value = Tensor(20, mindspore.float32)
        >>> input = Tensor(np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]), mindspore.float32)
        >>> output = ops.clamp(input, min_value, max_value)
        >>> print(output)
        [[ 5. 20.  5.  7.]
         [ 5. 11.  6. 20.]]
        >>> # case 2: the data type of input is list[Tensor]
        >>> min_value = 5
        >>> max_value = 20
        >>> input_x = Tensor(np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]), mindspore.float32)
        >>> input_y = Tensor(np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]), mindspore.float32)
        >>> output = ops.clamp([input_x,input_y], min_value, max_value)
        >>> for out in output:
        ...     print(out)
        [[ 5. 20.  5.  7.]
         [ 5. 11.  6. 20.]]
        [[ 5. 20.  5.  7.]
         [ 5. 11.  6. 20.]]
    """
    return clip_by_value(input, min, max)


def clip(input, min=None, max=None):
    r"""
    Alias for :func:`mindspore.ops.clamp` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return clamp(input, min, max)


@get_square_sum.register("Tensor")
def _get_square_sum(x):
    norm = P.ReduceSum(False)(F.square(x), ())
    norm = expand_dims(F.cast(norm, mstype.float32), 0)
    return norm


@apply_global_norm.register("Tensor", "Tensor", "Tensor")
def _apply_global_norm(clip_norm, global_norm, x):
    x_dtype = F.dtype(x)
    x = x * clip_norm / global_norm
    x = F.cast(x, x_dtype)
    return x


class _ClipByGlobalNorm(Cell):
    r"""
    Clips tensor values by the ratio of the sum of their norms.

    Args:
        clip_norm (Union(float, int)): The clipping ratio. Default: 1.0
        use_norm (Union(float, None)): The global norm. Default: ``None``

    Inputs:
        - **x** (Union(tuple[Tensor], list[Tensor])) - Input data to clip.

    Outputs:
        Tensor, a clipped Tensor.
    """

    def __init__(self, clip_norm=1.0, use_norm=None):
        """Initialize _ClipByGlobalNorm."""
        super(_ClipByGlobalNorm, self).__init__()
        # Add interface. This parameter is not used at present
        if use_norm is not None:
            raise ValueError(f"For '{self.cls_name}', input 'use_norm' only supports None currently, "
                             f"but got 'use_norm': {use_norm}")
        Validator.check_number("clip_norm", clip_norm, 0.0, Validator.GT, self.cls_name)
        self.clip_norm = Tensor([clip_norm], mstype.float32)
        self.hyper_map = C.HyperMap()
        self.greater_equal = P.GreaterEqual()

    def construct(self, x):
        square_sum = self.hyper_map(get_square_sum, x)
        global_norm = F.sqrt(F.addn(square_sum))
        cond = self.greater_equal(global_norm, self.clip_norm)
        global_norm = F.select(cond, global_norm, self.clip_norm)
        clip_x = self.hyper_map(F.partial(apply_global_norm, self.clip_norm, global_norm), x)
        return clip_x


@_primexpr
def _check_value(clip_norm):
    Validator.check_number("clip_norm", clip_norm, 0.0, Validator.GT, "clip_by_global_norm")


def clip_by_global_norm(x, clip_norm=1.0, use_norm=None):
    r"""
    Clips tensor values by the ratio of the sum of their norms.

    Note:
        - Input `x` should be a tuple or list of tensors. Otherwise, it will raise an error.
        - On the SEMI_AUTO_PARALLEL mode or AUTO_PARALLEL mode, if the input `x` is the gradient,
          the gradient norm values on all devices will be automatically aggregated by allreduce inserted after
          the local square sum of the gradients.

    Args:
        x (Union(tuple[Tensor], list[Tensor])): Input data to clip.
        clip_norm (Union(float, int)): The clipping ratio, it should be greater than 0. Default: ``1.0`` .
        use_norm (None): The global norm. Default: ``None`` . Currently only none is supported.

    Returns:
        tuple[Tensor], a clipped Tensor. It has the same data type as `x` and each Tensor in the output tuple is the
        same as the original input shape.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> import numpy as np
        >>> x1 = np.array([[2., 3.], [1., 2.]]).astype(np.float32)
        >>> x2 = np.array([[1., 4.], [3., 1.]]).astype(np.float32)
        >>> input_x = (Tensor(x1), Tensor(x2))
        >>> out = ops.clip_by_global_norm(input_x, 1.0)
        >>> print(out)
        (Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 2.98142403e-01,  4.47213590e-01],
         [ 1.49071202e-01,  2.98142403e-01]]), Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 1.49071202e-01,  5.96284807e-01],
         [ 4.47213590e-01,  1.49071202e-01]]))
    """

    _check_value(clip_norm)
    clip_val = _ClipByGlobalNorm(clip_norm, use_norm)(x)
    return clip_val
