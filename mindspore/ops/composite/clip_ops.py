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
def _check_shape(input_shape, out_shape):
    if input_shape != out_shape:
        raise ValueError("Cannot broadcast the shape of x to the shape of clip_value_min or clip_value_max.")


def clip_by_value(x, clip_value_min, clip_value_max):
    r"""
    Clips tensor values to a specified min and max.

    Limits the value of :math:`x` to a range, whose lower limit is 'clip_value_min'
    and upper limit is 'clip_value_max'.

    .. math::

        out_i= \left\{
        \begin{array}{align}
            clip\_value_{max} & \text{ if } x_i\ge  clip\_value_{max} \\
            x_i & \text{ if } clip\_value_{min} \lt x_i \lt clip\_value_{max} \\
            clip\_value_{min} & \text{ if } x_i \le clip\_value_{min} \\
        \end{array}\right.

    Note:
        'clip_value_min' needs to be less than or equal to 'clip_value_max'.

    Args:
          x (Tensor): Input data.
          clip_value_min (Tensor): The minimum value.
          clip_value_max (Tensor): The maximum value.

    Returns:
          Tensor, a clipped Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.ops import composite as C
        >>> import mindspore.common.dtype as mstype
        >>> min_value = Tensor(5, mstype.float32)
        >>> max_value = Tensor(20, mstype.float32)
        >>> x = Tensor(np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]), mstype.float32)
        >>> output = C.clip_by_value(x, min_value, max_value)
        >>> print(output)
        [[ 5. 20.  5.  7.]
         [ 5. 11.  6. 20.]]
    """
    min_op = P.Minimum()
    max_op = P.Maximum()
    x_min = min_op(x, clip_value_max)
    x_max = max_op(x_min, clip_value_min)
    _check_shape(F.shape(x), F.shape(x_max))
    return x_max


get_square_sum = C.MultitypeFuncGraph("get_square_sum")
@get_square_sum.register("Tensor")
def _get_square_sum(x):
    norm = P.ReduceSum(False)(F.square(x), ())
    norm = F.expand_dims(F.cast(norm, mstype.float32), 0)
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
        super(_ClipByGlobalNorm, self).__init__()
        # Add interface. This parameter is not used at present
        if use_norm is not None:
            raise ValueError("Input 'use_norm' only supports None currently!")
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
        Input 'x' should be a tuple or list of tensors. Otherwise, it will raise an error.

    Args:
          x (Union(tuple[Tensor], list[Tensor])): Input data to clip.
          clip_norm (Union(float, int)): The clipping ratio, it should be greater than 0. Default: 1.0
          use_norm (None): The global norm. Default: None. Currently only none is supported.

    Returns:
          tuple[Tensor], a clipped Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> x1 = np.array([[2., 3.], [1., 2.]]).astype(np.float32)
        >>> x2 = np.array([[1., 4.], [3., 1.]]).astype(np.float32)
        >>> input_x = (Tensor(x1), Tensor(x2))
        >>> out = clip_by_global_norm(input_x, 1.0)
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
