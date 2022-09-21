# Copyright 2020 Huawei Technologies Co., Ltd
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

"""Operations for clipping tensors to min/max values."""
from __future__ import absolute_import

import numpy as np
from mindspore.nn.cell import Cell
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype
from mindspore._checkparam import Rel
from mindspore._checkparam import Validator as validator
from mindspore.ops.primitive import constexpr


@constexpr
def _check_output_shape(input_shape, out_shape, prim_name=None):
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
    if input_shape != out_shape:
        raise ValueError(f"{msg_prefix} input 'x' shape must be equal to the output shape, but got "
                         f"input 'x' shape {input_shape}, output shape {out_shape}.")


def check_np_type(np_dtype, is_max_val):
    if not (np.issubsctype(np_dtype, np.floating) or np.issubsctype(np_dtype, np.integer) or
            np.issubsctype(np_dtype, np.complex64) or np.issubsctype(np_dtype, np.complex128) or
            np.issubsctype(np_dtype, np.bool_)):
        value_info = ("clip_value_max", "clip_value_min") if is_max_val else ("clip_value_min", "clip_value_max")
        raise ValueError(f"When {value_info[0]} is none, The date type of {value_info[1]} only support integer,"
                         f"floating, bool, complex64 or complex128. But got {np_dtype}")


@constexpr
def create_max_min_value(ms_type, is_max_val):
    """create max or min value"""
    np_dtype = mstype.dtype_to_nptype(ms_type)
    check_np_type(np_dtype, is_max_val)
    if np.issubsctype(np_dtype, np.floating):
        val = np.finfo(np_dtype).max if is_max_val else np.finfo(np_dtype).min
    elif np.issubsctype(np_dtype, np.integer):
        val = np.iinfo(np_dtype).max if is_max_val else np.iinfo(np_dtype).min
    elif np.issubsctype(np_dtype, np.complex64):
        val = np.finfo(np.float32).max if is_max_val else np.finfo(np.float32).min
        val = np.complex64(np.complex(val, val))
    elif np.issubsctype(np_dtype, np.complex128):
        val = np.finfo(np.float64).max if is_max_val else np.finfo(np.float64).min
        val = np.complex128(np.complex(val, val))
    else:
        val = np.bool_(True) if is_max_val else np.bool_(False)
    return Tensor(val, ms_type)


@constexpr
def raise_value_error():
    raise ValueError("At least one of 'clip_value_min' or 'clip_value_max' must not be None")


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
        `clip_value_min` needs to be less than or equal to `clip_value_max` . The data type of x, `clip_value_min` and
        `clip_value_max` should support implicit type conversion and cannot all be bool type.

    Args:
          x (Tensor): Input data. The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
          clip_value_min (Tensor): The minimum value. `clip_value_min` and `clip_value_max` cannot be all None.
                                   Default: None.
          clip_value_max (Tensor): The maximum value. `clip_value_min` and `clip_value_max` cannot be all None.
                                   Default: None.

    Returns:
          Tensor, a clipped Tensor. The data type is the one with higher precision or higher digits among
          the x, `clip_value_min` and `clip_value_max` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> import numpy as np
        >>> min_value = Tensor(5, mindspore.float32)
        >>> max_value = Tensor(20, mindspore.float32)
        >>> x = Tensor(np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]), mindspore.float32)
        >>> output = ops.clip_by_value(x, min_value, max_value)
        >>> print(output)
        [[ 5. 20.  5.  7.]
         [ 5. 11.  6. 20.]]
    """

    min_op = P.Minimum()
    max_op = P.Maximum()
    if clip_value_min is None and clip_value_max is None:
        raise_value_error()
    if clip_value_min is None:
        clip_value_min = create_max_min_value(F.dtype(clip_value_max), False)
    if clip_value_max is None:
        clip_value_max = create_max_min_value(F.dtype(clip_value_min), True)
    x_min = min_op(x, clip_value_max)
    x_max = max_op(x_min, clip_value_min)
    _check_output_shape(F.shape(x), F.shape(x_max), 'clip_by_value')
    return x_max


# The attribute grad_scale is needed for enabling the parallel mode
# If this is removed, c.clip_by_global_norm will have precision error in semi/auto parallel mode.
expand_dims = P.ExpandDims().add_prim_attr("grad_scale", True)


get_square_sum = C.MultitypeFuncGraph("get_square_sum")
@get_square_sum.register("Tensor")
def _get_square_sum(x):
    norm = P.ReduceSum(False)(F.square(x), ())
    norm = expand_dims(F.cast(norm, mstype.float32), 0)
    return norm


apply_global_norm = C.MultitypeFuncGraph("apply_global_norm")
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
        use_norm (Union(float, None)): The global norm. Default: None

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
        validator.check_number("clip_norm", clip_norm, 0.0, Rel.GT, self.cls_name)
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


@constexpr
def _check_value(clip_norm):
    validator.check_number("clip_norm", clip_norm, 0.0, Rel.GT, "clip_by_global_norm")
    return clip_norm


def clip_by_global_norm(x, clip_norm=1.0, use_norm=None):
    r"""
    Clips tensor values by the ratio of the sum of their norms.

    Note:
        Input `x` should be a tuple or list of tensors. Otherwise, it will raise an error.

    Note:
        On the SEMI_AUTO_PARALLEL mode or AUTO_PARALLEL mode, if the input `x` is the gradient,
        the gradient norm values on all devices will be automatically aggregated by allreduce inserted after the local
        square sum of the gradients.

    Args:
        x (Union(tuple[Tensor], list[Tensor])): Input data to clip.
          The shape of each Tensor in tuple is :math:`(N,*)` where :math:`*` means,
          any number of additional dimensions.
        clip_norm (Union(float, int)): The clipping ratio, it should be greater than 0. Default: 1.0
        use_norm (None): The global norm. Default: None. Currently only none is supported.

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

    clip_norm = _check_value(clip_norm)
    clip_val = _ClipByGlobalNorm(clip_norm, use_norm)(x)
    return clip_val
