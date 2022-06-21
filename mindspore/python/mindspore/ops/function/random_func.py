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

from mindspore.ops import operations as P
from ...common import dtype as mstype
from .._primitive_cache import _get_cache_prim


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



def random_categorical(x, num_sample, seed=0, dtype=mstype.int64):
    r"""
    Generates random samples from a given categorical distribution tensor.

    Args:
        dtype (mindspore.dtype): The type of output. Its value must be one of mindspore.int16,
            mindspore.int32 and mindspore.int64. Default: mindspore.int64.

    Inputs:
        - **logits** (Tensor) - The input tensor. 2-D Tensor with shape [batch_size, num_classes].
        - **num_sample** (int) - Number of sample to be drawn. Only constant values is allowed.
        - **seed** (int) - Random seed. Default: 0. Only constant values is allowed.

    Outputs:
        - **output** (Tensor) - The output Tensor with shape [batch_size, num_samples].

    Raises:
        TypeError: If `dtype` is not one of the following: mindspore.int16, mindspore.int32, mindspore.int64.
        TypeError: If `logits` is not a Tensor.
        TypeError: If neither `num_sample` nor `seed` is an int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore.ops import functional as F
        >>> x = np.random.random((10, 5)).astype(np.float32)
        >>> net = F.random_categorical(x, 8)
        >>> output = net(Tensor(x))
        >>> result = output.shape
        >>> print(result)F
        (10, 8)
    """
    random_categorical_ = P.RandomCategorical(dtype)
    return random_categorical_(x, num_sample, seed)


__all__ = [
    'standard_laplace',
    'random_categorical'
]
__all__.sort()
