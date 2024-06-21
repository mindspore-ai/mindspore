# Copyright 2020-2024 Huawei Technologies Co., Ltd
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
"""normalization"""
from __future__ import absolute_import
from __future__ import division

from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.common import dtype as mstype
from mindspore.nn.cell import Cell

__all__ = ['LayerNorm']


class LayerNorm(Cell):
    r"""
    Applies Layer Normalization over a mini-batch of inputs.

    Layer Normalization is widely used in recurrent neural networks. It applies
    normalization on a mini-batch of inputs for each single training case as described
    in the paper `Layer Normalization <https://arxiv.org/pdf/1607.06450.pdf>`_. Unlike Batch
    Normalization, Layer Normalization performs exactly the same computation at training and
    testing time. It is applied across all channels and pixel but only one batch size.
    :math:`\gamma` and :math:`\beta` are trainable scale and shift.
    It can be described using the following formula:

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Args:
        normalized_shape (Union(tuple[int], list[int])): The normalized shape of `x` for LayerNorm
        gamma_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the :math:`\gamma` weight.
            The values of str refer to the function `initializer` including ``'zeros'`` , ``'ones'`` ,
            ``'xavier_uniform'`` , ``'he_uniform'`` , etc. Default: ``'ones'`` .
        beta_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the :math:`\beta` weight.
            The values of str refer to the function `initializer` including ``'zeros'`` , ``'ones'`` ,
            ``'xavier_uniform'`` , ``'he_uniform'`` , etc. Default: ``'zeros'`` .
        eps (float): A value added to the denominator for numerical stability(:math:`\epsilon`). Default: ``1e-5`` .
        elementwise_affine (bool): A bool value, When set to True, gamma and beta can be learned. Default: True.
        dtype (:class:`mindspore.dtype`): Dtype of Parameters. Default: ``mstype.float32`` .

    Inputs:
        - **x** (Tensor) - The shape is :math:`(N, *)`, where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, the normalized and scaled offset tensor, has the same shape and data type as the `x`.

    Raises:
        TypeError: If `epsilon` is not a float.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> x = ms.Tensor(np.ones([20, 5, 10, 10]), ms.float32)
        >>> shape1 = x.shape[1:]
        >>> m = ms.nn.extend.LayerNorm(shape1)
        >>> output = m(x).shape
        >>> print(output)
        (20, 5, 10, 10)
    """

    def __init__(self,
                 normalized_shape,
                 gamma_init='ones',
                 beta_init='zeros',
                 eps=1e-5,
                 elementwise_affine=True,
                 dtype=mstype.float32
                 ):
        """Initialize LayerNorm."""
        super(LayerNorm, self).__init__()
        if not isinstance(normalized_shape, (tuple, list)):
            raise TypeError(f"For '{self.cls_name}', the type of 'normalized_shape' must be tuple[int] or list[int], "
                            f"but got {normalized_shape} and the type is {type(normalized_shape)}.")
        if not normalized_shape:
            raise ValueError(
                f"Expected normalized_shape to be at least 1-dimensional, i.e., containing at "
                f"least one element, but got normalized_shape = {normalized_shape}"
            )
        self.normalized_shape = normalized_shape
        self.epsilon = eps
        self.gamma = Parameter(initializer(
            gamma_init, normalized_shape, dtype=dtype), name="gamma", requires_grad=elementwise_affine)
        self.beta = Parameter(initializer(
            beta_init, normalized_shape, dtype=dtype), name="beta", requires_grad=elementwise_affine)

    def construct(self, input_x):
        y = F.layer_norm(input_x, self.normalized_shape, self.gamma.astype(input_x.dtype),
                         self.beta.astype(input_x.dtype), self.epsilon)
        return y

    def extend_repr(self):
        return 'normalized_shape={}, gamma{}, beta={}'.format(self.normalized_shape, self.gamma, self.beta)
