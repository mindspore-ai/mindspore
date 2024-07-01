# Copyright 2024 Huawei Technologies Co., Ltd
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

"""basic"""
from __future__ import absolute_import

import math

import mindspore.common.dtype as mstype
from mindspore import _checkparam as Validator
from mindspore._extends import cell_attr_register
from mindspore.common.initializer import initializer, HeUniform, Uniform
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.nn.cell import Cell
from mindspore.ops import operations as P

__all__ = ['Linear']


class Linear(Cell):
    r"""
    The linear connected layer.

    Applies linear connected layer for the input. This layer implements the operation as:

    .. math::
        \text{outputs} = X * kernel + bias

    where :math:`X` is the input tensors, :math:`\text{kernel}` is a weight matrix with the same
    data type as the :math:`X` created by the layer, and :math:`\text{bias}` is a bias vector
    with the same data type as the :math:`X` created by the layer (only if has_bias is True).

    Args:
        in_features (int): The number of features in the input space.
        out_features (int): The number of features in the output space.
        bias (bool): Specifies whether the layer uses a bias vector :math:`\text{bias}`. Default: ``True``.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `x`. The values of str refer to the function `initializer`. Default: ``None`` ,
            weight will be initialized using HeUniform.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as `x`. The values of str refer to the function `initializer`. Default: ``None`` ,
            bias will be initialized using Uniform.
        dtype (:class:`mindspore.dtype`): Data type of Parameter. Default: ``None`` .

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(*, in\_features)`. The `in_features` in `Args` should be equal
          to :math:`in\_features` in `Inputs`.

    Outputs:
        Tensor of shape :math:`(*, out\_features)`.

    Raises:
        TypeError: If `in_features` or `out_features` is not an int.
        TypeError: If `bias` is not a bool.
        ValueError: If length of shape of `weight_init` is not equal to 2 or shape[0] of `weight_init`
                    is not equal to `out_features` or shape[1] of `weight_init` is not equal to `in_features`.
        ValueError: If length of shape of `bias_init` is not equal to 1
                    or shape[0] of `bias_init` is not equal to `out_features`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore import nn
        >>> import numpy as np
        >>> x = Tensor(np.array([[180, 234, 154], [244, 48, 247]]), mindspore.float32)
        >>> net = nn.extend.Linear(3, 4)
        >>> output = net(x)
        >>> print(output.shape)
        (2, 4)
    """

    @cell_attr_register(attrs=['has_bias'])
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 weight_init=None,
                 bias_init=None,
                 dtype=None):
        """Initialize Linear."""
        super(Linear, self).__init__()
        self.in_features = Validator.check_positive_int(
            in_features, "in_features", self.cls_name)
        self.out_features = Validator.check_positive_int(
            out_features, "out_features", self.cls_name)
        self.has_bias = Validator.check_bool(
            bias, "has_bias", self.cls_name)
        self.dense = P.Dense()
        if dtype is None:
            dtype = mstype.float32
        if isinstance(weight_init, Tensor):
            if weight_init.ndim != 2 or weight_init.shape[0] != out_features or \
                    weight_init.shape[1] != in_features:
                raise ValueError(f"For '{self.cls_name}', weight init shape error. The ndim of 'weight_init' must "
                                 f"be equal to 2, and the first dim must be equal to 'out_features', and the "
                                 f"second dim must be equal to 'in_features'. But got 'weight_init': {weight_init}, "
                                 f"'out_features': {out_features}, 'in_features': {in_features}.")
        if weight_init is None:
            weight_init = HeUniform(math.sqrt(5))
        self.weight = Parameter(initializer(
            weight_init, [out_features, in_features], dtype=dtype), name="weight")

        self.bias = None
        if self.has_bias:
            if isinstance(bias_init, Tensor):
                if bias_init.ndim != 1 or bias_init.shape[0] != out_features:
                    raise ValueError(f"For '{self.cls_name}', bias init shape error. The ndim of 'bias_init' must "
                                     f"be equal to 1, and the first dim must be equal to 'out_features'. But got "
                                     f"'bias_init': {bias_init}, 'out_features': {out_features}.")
            if bias_init is None:
                bound = 1 / math.sqrt(in_features)
                bias_init = Uniform(scale=bound)
            self.bias = Parameter(initializer(
                bias_init, [out_features], dtype=dtype), name="bias")

    def construct(self, x):
        x = self.dense(x, self.weight, self.bias)
        return x

    def extend_repr(self):
        s = f'input_features={self.in_features}, output_features={self.out_features}'
        if self.has_bias:
            s += f', has_bias={self.has_bias}'
        return s
