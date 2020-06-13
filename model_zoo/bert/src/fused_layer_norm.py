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
"""fused layernorm"""
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.ops.primitive import constexpr
import mindspore.common.dtype as mstype
from mindspore.nn.cell import Cell

import numpy as np


__all__ = ['FusedLayerNorm']

@constexpr
def get_shape_for_norm(x_shape, begin_norm_axis):
    print("input_shape: ", x_shape)
    norm_shape = x_shape[begin_norm_axis:]
    output_shape = (1, -1, 1, int(np.prod(norm_shape)))
    print("output_shape: ", output_shape)
    return output_shape

class FusedLayerNorm(Cell):
    r"""
    Applies Layer Normalization over a mini-batch of inputs.

    Layer normalization is widely used in recurrent neural networks. It applies
    normalization over a mini-batch of inputs for each single training case as described
    in the paper `Layer Normalization <https://arxiv.org/pdf/1607.06450.pdf>`_. Unlike batch
    normalization, layer normalization performs exactly the same computation at training and
    testing times. It can be described using the following formula. It is applied across all channels
    and pixel but only one batch size.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Args:
        normalized_shape (Union(tuple[int], list[int]): The normalization is performed over axis
            `begin_norm_axis ... R - 1`.
        begin_norm_axis (int): It first normalization dimension: normalization will be performed along dimensions
            `begin_norm_axis: rank(inputs)`, the value should be in [-1, rank(input)). Default: -1.
        begin_params_axis (int): The first parameter(beta, gamma)dimension: scale and centering parameters
            will have dimensions `begin_params_axis: rank(inputs)` and will be broadcast with
            the normalized inputs accordingly, the value should be in [-1, rank(input)). Default: -1.
        gamma_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the gamma weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', 'xavier_uniform',
            'he_uniform', etc. Default: 'ones'.
        beta_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the beta weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', 'xavier_uniform',
            'he_uniform', etc. Default: 'zeros'.
        use_batch_nrom (bool): Whether use batchnorm to preocess.

    Inputs:
        - **input_x** (Tensor) - The shape of 'input_x' is :math:`(x_1, x_2, ..., x_R)`,
          and `input_shape[begin_norm_axis:]` is equal to `normalized_shape`.

    Outputs:
        Tensor, the normalized and scaled offset tensor, has the same shape and data type as the `input_x`.

    Examples:
        >>> x = Tensor(np.ones([20, 5, 10, 10]), mindspore.float32)
        >>> shape1 = x.shape[1:]
        >>> m = nn.LayerNorm(shape1,  begin_norm_axis=1, begin_params_axis=1)
        >>> m(x)
    """
    def __init__(self,
                 normalized_shape,
                 begin_norm_axis=-1,
                 begin_params_axis=-1,
                 gamma_init='ones',
                 beta_init='zeros',
                 use_batch_norm=False):
        super(FusedLayerNorm, self).__init__()
        if not isinstance(normalized_shape, (tuple, list)):
            raise TypeError("The type of 'normalized_shape' should be tuple[int] or list[int], but '{}' type is {}."
                            .format(normalized_shape, type(normalized_shape)))
        self.normalized_shape = normalized_shape
        self.begin_norm_axis = begin_norm_axis
        self.begin_params_axis = begin_params_axis
        self.gamma = Parameter(initializer(
            gamma_init, normalized_shape), name="gamma")
        self.beta = Parameter(initializer(
            beta_init, normalized_shape), name="beta")
        self.layer_norm = P.LayerNorm(begin_norm_axis=self.begin_norm_axis, begin_params_axis=self.begin_params_axis)

        self.batch_norm = P.BatchNorm(is_training=True, epsilon=1e-5)
        self.use_batch_norm = use_batch_norm

    def construct(self, input_x):
        if self.use_batch_norm and self.training:
            ones = P.Fill()(mstype.float32, F.shape(input_x)[:self.begin_norm_axis], 1.0)
            zeros = P.Fill()(mstype.float32, F.shape(input_x)[:self.begin_norm_axis], 0.0)
            shape_x = F.shape(input_x)
            norm_shape = get_shape_for_norm(shape_x, self.begin_norm_axis)
            input_x = F.reshape(input_x, norm_shape)
            output, _, _, _, _, _ = self.batch_norm(input_x, ones, zeros, None, None)
            output = F.reshape(output, shape_x)
            y = output * self.gamma + self.beta
        else:
            y, _, _ = self.layer_norm(input_x, self.gamma, self.beta)
        return y

    def extend_repr(self):
        """Display instance object as string."""
        s = 'normalized_shape={}, begin_norm_axis={}, begin_params_axis={}, gamma{}, beta={}'.format(
            self.normalized_shape, self.begin_norm_axis, self.begin_params_axis, self.gamma, self.beta)
        return s
