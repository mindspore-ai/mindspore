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

"""Defines parameter operators with functional form."""

import numpy as np
from mindspore.ops import operations as P
from mindspore.ops.primitive import constexpr
from ...common.seed import _get_graph_seed
from ...common.tensor import Tensor
from .._primitive_cache import _get_cache_prim
from .._utils import get_broadcast_shape


def random_gamma(shape, alpha, seed=0, seed2=0):
    r"""
    Outputs random values from the Gamma distribution(s) described by alpha.


    Args:
        shape (Tensor): The shape of random tensor to be generated.
            Must be one of the following types: int32, int64. 1-D integer tensor.
        alpha (Tensor): The alpha α distribution parameter.
            A Tensor. Must be one of the following types: half, float32, float64.
        seed (int): Seed is used as entropy source for the random number engines to generate
            pseudo-random numbers, must be non-negative. Default: None, which will be treated as 0.

    Returns:
        Tensor. The shape should be equal to the concat shape between the input `shape` and the broadcast
        of `alpha`.
        The dtype is the same type as alpha.

    Raises:
        TypeError: If `shape` is not a Tensor.
        TypeError: If `alpha` is not a Tensor.
        TypeError: If `seed` is not an int.
        TypeError: If dtype of `alpha` is not half, float32 or float64.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore.ops import functional as F
        >>> shape = Tensor(np.array([7, 5]), mindspore.int32)
        >>> alpha = Tensor(np.array([0.5, 1.5]), mindspore.float32)
        >>> output = F.random_gamma(shape, alpha, seed=5)
        >>> result = output.shape
        >>> print(result)
        (7, 5, 2)
    """

    alpha_type = P.DType()(alpha)
    beta = Tensor(np.array([1.0]), alpha_type)
    alpha_shape = P.Shape()(alpha)
    beta_shape = P.Shape()(beta)
    broadcast_shape = get_broadcast_shape(alpha_shape, beta_shape, "random_gamma",
                                          arg_name1="alpha", arg_name2="beta")
    broadcast_shape_t = tuple(broadcast_shape)
    broadcast_to = P.BroadcastTo(broadcast_shape_t)
    alpha_broadcast = broadcast_to(alpha)
    random_gamma_op = _get_cache_prim(P.RandomGamma)(seed=seed, seed2=seed2)
    output = random_gamma_op(shape, alpha_broadcast)

    return output


@constexpr(reuse_result=False)
def _get_seed(op_seed, kernel_name):
    "Get the graph-level seed."
    return _get_graph_seed(op_seed, kernel_name)


def standard_laplace(shape, seed=0, seed2=0):
    r"""
    Generates random numbers according to the Laplace random number distribution (mean=0, lambda=1).
    It is defined as:

    .. math::
        \text{f}(x) = \frac{1}{2}\exp(-|x|),

    Args:
        shape (tuple): The shape of random tensor to be generated. Only constant value is allowed.
        seed (int): Random seed. Default: 0.
        seed2 (int): Random seed2. Default: 0.

    Returns:
        Tensor. The shape that the input 'shape' denotes. The dtype is float32.

    Raises:
        TypeError: If neither seed nor seed2 is an int.
        TypeError: If shape is not a tuple.
        ValueError: If shape is not a constant value.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> from mindspore.ops import functional as F
        >>> shape = (4, 4)
        >>> output = F.standard_laplace(shape)
        >>> result = output.shape
        >>> print(result)
        (4, 4)
    """
    standard_laplace_op = _get_cache_prim(P.StandardLaplace)(seed=seed, seed2=seed2)
    output = standard_laplace_op(shape)
    return output


def standard_normal(shape, seed=0, seed2=0):
    r"""
    Generates random numbers according to the standard Normal (or Gaussian) random number distribution.

    Returns the tensor with the given shape, the random numbers in it drawn from normal distributions
    whose mean is 0 and standard deviation is 1.

    .. math::
        f(x)=\frac{1}{\sqrt{2 \pi}} e^{\left(-\frac{x^{2}}{2}\right)}

    Args:
        shape (tuple): The shape of random tensor to be generated. Only constant value is allowed.
        seed (int): Random seed, must be non-negative. Default: 0.
        seed2 (int): Random seed2, must be non-negative. A second seed to avoid seed collision. Default: 0.

    Returns:
        Tensor. The shape is the same as the input `shape`. The dtype is float32.

    Raises:
        TypeError: If neither `seed` nor `seed2` is an int.
        TypeError: If `shape` is not a tuple.
        ValueError: If `shape` is not a constant value.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore.ops import functional as F
        >>> shape = (4, 4)
        >>> output = F.standard_normal(shape)
        >>> result = output.shape
        >>> print(result)
        (4, 4)
    """
    standard_normal_op = _get_cache_prim(P.StandardNormal)(seed=seed, seed2=seed2)
    return standard_normal_op(shape)


__all__ = [
    'standard_laplace',
    'standard_normal',
]
__all__.sort()
