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

"""Operators for clip."""

from collections.abc import Iterable
from ..._checkparam import Validator as validator
from ..._checkparam import Rel
from ...common import dtype as mstype
from ..primitive import PrimitiveWithInfer, prim_attr_register


class ClipByNorm(PrimitiveWithInfer):
    r"""
    Clips tensor values to a maximum :math:`L_2`-norm.

    Note:
        The output tensor of this operator remains the same with input tensor if the :math:`L_2`-norm of the input
        tensor is not greater than the argument `clip_norm`. Otherwise the output tensor will be normalized as:

        .. math::
            \text{output}(X) = \frac{\text{clip_norm} * X}{L_2(X)},

        where :math:`L_2(X)` is the :math:`L_2`-norm of :math:`X`.

    Args:
        axis (Union[None, int, tuple(int), list(int)]): Compute the `L_2`-norm along the specific dimension.
                                                       Default: None, all dimensions to calculate.

    Inputs:
        - **x** (Tensor) - Tensor of shape N-D. The type must be float16 or float32.
        - **clip_norm** (Tensor) - A scalar Tensor of shape :math:`()` or :math:`(1)`.
          Or a Tensor which shape can be broadcast to the shape of `x`. The type must be float16 or float32.

    Outputs:
        Tensor, clipped Tensor with the same shape as the `x`, whose type is float32.

    Raises:
        TypeError: If `axis` is not one of None, int, tuple(int) and list(int).
        TypeError: If dtype of `x` is neither float16 nor float32.
        TypeError: If dtype of `clip_norm` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore.ops.operations import clip_ops
        >>> clip_by_norm = clip_ops.ClipByNorm()
        >>> x = Tensor(np.random.randint(0, 10, [4, 16]), mindspore.float32)
        >>> clip_norm = Tensor(np.array([100]).astype(np.float32))
        >>> output = clip_by_norm(x, clip_norm)
        >>> print(output.shape)
        (4, 16)
    """

    @prim_attr_register
    def __init__(self, axis=None):
        """Initialize ClipByNorm"""
        self.axis = () if axis is None else axis
        validator.check_value_type('axis', self.axis, [int, tuple, list], self.name)
        axis_check = self.axis if isinstance(self.axis, Iterable) else (self.axis,)
        for i, value in enumerate(axis_check):
            validator.check_value_type('axis[%d]' % i, value, [int], self.name)
        self.init_attrs['axis'] = self.axis
        self.add_prim_attr('axis', self.axis)
        self.init_prim_io_names(inputs=['x', 'clip_norm'], outputs=['output'])

    def infer_shape(self, x_shape, clip_norm_shape):
        """Infer shape for ClipByNorm"""
        x_dim = len(x_shape)
        axis = self.axis if isinstance(self.axis, Iterable) else (self.axis,)
        for _, value in enumerate(axis):
            validator.check_int_range(value, -x_dim, x_dim, Rel.INC_LEFT, 'axis', self.name)
        return x_shape

    def infer_dtype(self, x_type, clip_norm_type):
        """Infer data type for ClipByNorm"""
        validator.check_tensor_dtype_valid("x_type", x_type, [mstype.float16, mstype.float32], self.name)
        validator.check_tensor_dtype_valid("clip_norm_type", clip_norm_type,
                                           [mstype.float16, mstype.float32], self.name)
        return mstype.float32
