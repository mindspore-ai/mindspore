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
from mindspore.ops import functional as F
from mindspore.ops.primitive import constexpr
from mindspore.ops.composite.multitype_ops import _constexpr_utils as const_utils
from ...common import dtype as mstype
from ...common.seed import _get_graph_seed
from ...common.tensor import Tensor
from ..operations.random_ops import RandomShuffle, RandomChoiceWithMask
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
        seed2 (int): Seed2 is used as entropy source for the random number engines to generate
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
        >>> from mindspore import ops
        >>> shape = Tensor(np.array([7, 5]), mindspore.int32)
        >>> alpha = Tensor(np.array([0.5, 1.5]), mindspore.float32)
        >>> output = ops.random_gamma(shape, alpha, seed=5)
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
        shape (Union[tuple, Tensor]): The shape of random tensor to be generated. Only constant value is allowed
          when the input type is tuple. And the operator supports dynamic shape only when the input type is Tensor.
        seed (int): Random seed. Default: 0.
        seed2 (int): Random seed2. Default: 0.

    Returns:
        Tensor. The shape that the input 'shape' denotes. The dtype is float32.

    Raises:
        TypeError: If seed or seed2 is not an int.
        TypeError: If shape is neither a tuple nor a Tensor.
        ValueError: If seed or seed2 is not a non-negative int.
        ValueError: If shape is a tuple containing non-positive items.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> from mindspore import ops
        >>> shape = (4, 4)
        >>> output = ops.standard_laplace(shape)
        >>> result = output.shape
        >>> print(result)
        (4, 4)
    """
    standard_laplace_op = _get_cache_prim(P.StandardLaplace)(seed=seed, seed2=seed2)
    output = standard_laplace_op(shape)
    return output


def random_categorical(logits, num_sample, seed=0, dtype=mstype.int64):
    r"""
    Generates random samples from a given categorical distribution tensor.

    Args:
        logits (Tensor): The input tensor. 2-D Tensor with shape :math:`(batch\_size, num\_classes)`.
        num_sample (int):  Number of sample to be drawn. Only constant values is allowed.
        seed (int):  Random seed. Only constant values is allowed. Default: 0.
        dtype (mindspore.dtype): The type of output. Its value must be one of mindspore.int16,
            mindspore.int32 and mindspore.int64. Default: mindspore.int64.

    Returns:
        Tensor, The output Tensor with shape :math:`(batch\_size, num\_samples)`.

    Raises:
        TypeError: If `dtype` is not one of the following: mindspore.int16, mindspore.int32, mindspore.int64.
        TypeError: If `logits` is not a Tensor.
        TypeError: If neither `num_sample` nor `seed` is an int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import ops
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> import numpy as np
        >>> logits = Tensor(np.random.random((10, 5)).astype(np.float32), mstype.float32)
        >>> net = ops.random_categorical(logits, 8)
        >>> result = net.shape
        >>> print(result)
        (10, 8)
    """
    random_categorical_ = P.RandomCategorical(dtype)
    return random_categorical_(logits, num_sample, seed)


def uniform(shape, minval, maxval, seed=None, dtype=mstype.float32):
    """
    Generates random numbers according to the Uniform random number distribution.

    Note:
        The number in tensor minval should be strictly less than maxval at any position after broadcasting.

    Args:
        shape (tuple): The shape of random tensor to be generated.
          The format is :math:`(N,*)` where :math:`*` means, any number of additional dimensions
          and the length of :math:`(N,*)` should be less than 8 in broadcast operation.
        minval (Tensor): The distribution parameter `a`.
          It defines the minimum possible generated value, with int32 or float32 data type.
          If dtype is int32, only one number is allowed.
        maxval (Tensor): The distribution parameter `b`.
          It defines the maximum possible generated value, with int32 or float32 data type.
          If dtype is int32, only one number is allowed.
        seed (int): Seed is used as entropy source for the random number engines to generate pseudo-random numbers,
          must be non-negative. Default: None, which will be treated as 0.
        dtype (mindspore.dtype): Type of the Uniform distribution. If it is int32, it generates numbers from discrete
          uniform distribution; if it is float32, it generates numbers from continuous uniform distribution. It only
          supports these two data types. Default: mindspore.float32.

    Returns:
        Tensor. The shape should be equal to the broadcasted shape between the input `shape` and shapes
        of `minval` and `maxval`.
        The dtype is designated as the input `dtype`.

    Raises:
        TypeError: If `shape` is not tuple.
        TypeError: If 'minval' or 'maxval' is neither int32 nor float32
            and dtype of 'minval' is not the same as 'maxval'.
        TypeError: If `seed` is not an int.
        TypeError: If 'dtype' is neither int32 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> import mindspore
        >>> import numpy as np
        >>> # For discrete uniform distribution, only one number is allowed for both minval and maxval:
        >>> shape = (4, 2)
        >>> minval = Tensor(1, mindspore.int32)
        >>> maxval = Tensor(2, mindspore.int32)
        >>> output = ops.uniform(shape, minval, maxval, seed=5, dtype=mindspore.int32)
        >>>
        >>> # For continuous uniform distribution, minval and maxval can be multi-dimentional:
        >>> shape = (3, 1, 2)
        >>> minval = Tensor(np.array([[3, 4], [5, 6]]), mindspore.float32)
        >>> maxval = Tensor([8.0, 10.0], mindspore.float32)
        >>> output = ops.uniform(shape, minval, maxval, seed=5)
        >>> result = output.shape
        >>> print(result)
        (3, 2, 2)
    """
    if not isinstance(minval, Tensor) or not isinstance(maxval, Tensor):
        raise TypeError(f"For functional operator[uniform], the input[minval] and input[maxval] must be a Tensor.")

    minval_dtype = F.dtype(minval)
    maxval_dtype = F.dtype(maxval)
    const_utils.check_type_valid(dtype, [mstype.int32, mstype.float32], 'uniform')
    const_utils.check_tensors_dtype_same(minval_dtype, dtype, "uniform")
    const_utils.check_tensors_dtype_same(maxval_dtype, dtype, "uniform")
    seed1, seed2 = _get_seed(seed, "uniform")
    if const_utils.is_same_type(dtype, mstype.int32):
        random_uniform = P.UniformInt(seed1, seed2)
        value = random_uniform(shape, minval, maxval)
    else:
        uniform_real = P.UniformReal(seed1, seed2)
        random_uniform = uniform_real(shape)
        value = random_uniform * (maxval - minval) + minval
    return value


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
        TypeError: If `seed` or `seed2` is not an int.
        TypeError: If `shape` is not a tuple.
        ValueError: If `shape` is not a constant value.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import ops
        >>> shape = (4, 4)
        >>> output = ops.standard_normal(shape)
        >>> result = output.shape
        >>> print(result)
        (4, 4)
    """
    standard_normal_op = _get_cache_prim(P.StandardNormal)(seed=seed, seed2=seed2)
    return standard_normal_op(shape)


def uniform_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, seed=0,
                              remove_accidental_hits=False):
    r"""
    Uniform candidate sampler.

    This function samples a set of classes(sampled_candidates) from [0, range_max-1] based on uniform distribution.
    If unique=True, candidates are drawn without replacement, else unique=False with replacement.

    Args:
        true_classes (Tensor): A Tensor. The target classes with a Tensor shape of (batch_size, num_true).
        num_true (int): The number of target classes in each training example.
        num_sampled (int): The number of classes to randomly sample. The sampled_candidates will have a shape
            of num_sampled. If unique=True, num_sampled must be less than or equal to range_max.
        unique (bool): Whether all sampled classes in a batch are unique.
        range_max (int): The number of possible classes, must be positive.
        seed (int): Used for random number generation, must be non-negative. If seed has a value of 0,
            the seed will be replaced with a randomly generated value. Default: 0.
        remove_accidental_hits (bool): Whether accidental hit is removed. Default: False.

    Returns:
        - **sampled_candidates** (Tensor) - The sampled_candidates is independent of the true classes.
          Shape: (num_sampled, ).
        - **true_expected_count** (Tensor) - The expected counts under the sampling distribution of each
          of true_classes. Shape: (batch_size, num_true).
        - **sampled_expected_count** (Tensor) - The expected counts under the sampling distribution of
          each of sampled_candidates. Shape: (num_sampled, ).

    Raises:
        TypeError: If neither `num_true` nor `num_sampled` is an int.
        TypeError: If neither `unique` nor `remove_accidental_hits` is a bool.
        TypeError: If neither `range_max` nor `seed` is an int.
        TypeError: If `true_classes` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> data = Tensor(np.array([[1], [3], [4], [6], [3]], dtype=np.int32))
        >>> output1, output2, output3 = ops.uniform_candidate_sampler(data, 1, 3, False, 4, 1)
        >>> print(output1.shape)
        (3,)
        >>> print(output2.shape)
        (5, 1)
        >>> print(output3.shape)
        (3,)
    """
    sampler_op = _get_cache_prim(P.UniformCandidateSampler)(num_true, num_sampled, unique, range_max, seed=seed,
                                                            remove_accidental_hits=remove_accidental_hits)
    sampled_candidates, true_expected_count, sampled_expected_count = sampler_op(true_classes)
    return sampled_candidates, true_expected_count, sampled_expected_count


def random_poisson(shape, rate, seed=None, dtype=mstype.float32):
    r"""
    Generates random numbers according to the Poisson random number distribution.

    .. math::

        \text{P}(i|μ) = \frac{\exp(-μ)μ^{i}}{i!}

    Args:
        shape (Tensor): The shape of random tensor to be sampled from each poisson distribution, 1-D `Tensor` whose
          dtype is mindspore.dtype.int32 or mindspore.dtype.int64.
        rate (Tensor): The μ parameter the distribution was constructed with. It represents the mean of the distribution
          and also the variance of the distribution. It should be a `Tensor` whose dtype is mindspore.dtype.int64,
          mindspore.dtype.int32, mindspore.dtype.float64, mindspore.dtype.float32 or mindspore.dtype.float16.
        seed (int): Seed is used as entropy source for the random number engines to generate pseudo-random numbers
          and must be non-negative. Default: None, which will be treated as 0.
        dtype (mindspore.dtype): The data type of output: mindspore.dtype.int64, mindspore.dtype.int32,
          mindspore.dtype.float64, mindspore.dtype.float32 or mindspore.dtype.float16. Default: mindspore.dtype.float32.

    Returns:
        A Tensor whose shape is `mindspore.concat(['shape', mindspore.shape('rate')], axis=0)` and data type is equal to
        argument `dtype`.

    Raises:
        TypeError: If `shape` is not a Tensor.
        TypeError: If datatype of `shape` is not mindspore.dtype.int64 nor mindspore.dtype.int32.
        ValueError: If shape of `shape` is not 1-D.
        TypeError: If `rate` is not a Tensor nor a scalar.
        TypeError: If datatype of `rate` is not in [mindspore.dtype.int64, mindspore.dtype.int32,
          mindspore.dtype.float64, mindspore.dtype.float32 or mindspore.dtype.float16].
        TypeError: If `seed` is not a non-negtive int.
        TypeError: If `dtype` is not in [mindspore.dtype.int64, mindspore.dtype.int32, mindspore.dtype.float64,
          mindspore.dtype.float32 nor mindspore.dtype.float16].
        ValueError: If any element of input `shape` tensor is not positive.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> import mindspore
        >>> # case 1: 1-D shape, 2-D rate, float64 output
        >>> shape = Tensor(np.array([2, 2]), mindspore.int64)
        >>> rate = Tensor(np.array([[5.0, 10.0], [5.0, 1.0]]), mindspore.float32)
        >>> output = ops.random_poisson(shape, rate, seed=5, dtype=mindspore.float64)
        >>> print(output.shape, output.dtype)
        (2, 2, 2, 2) float64
        >>> # case 2: 1-D shape, scalar rate, int64 output
        >>> shape = Tensor(np.array([2, 2]), mindspore.int64)
        >>> rate = Tensor(5.0, mindspore.float64)
        >>> output = ops.random_poisson(shape, rate, seed=5, dtype=mindspore.int64)
        >>> print(output.shape, output.dtype)
        (2, 2) Int64
    """
    seed1, seed2 = _get_seed(seed, "random_poisson")
    prim_random_poisson = P.random_ops.RandomPoisson(seed1, seed2, dtype)
    value = prim_random_poisson(shape, rate)
    return value


def random_shuffle(x, seed=0, seed2=0):
    r"""
    Randomly shuffles a Tensor along its first dimension.

    Args:
        x (Tensor): The Tensor need be shuffled.
        seed (int): The operator-level random seed, used to generate random numbers, must be non-negative. Default: 0.
        seed2 (int): The global random seed and it will combile with the operator-level random seed to determine the
            final generated random number, must be non-negative. Default: 0.

    Returns:
        Tensor. The shape and type are the same as the input `x`.

    Raises:
        TypeError: If data type of `seed` or `seed2` is not int.

    Supported Platforms:
        ``CPU`` ``GPU``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3, 4]), mstype.float32)
        >>> shuffle = ops.RandomShuffle(seed=1, seed2=1)
        >>> output = shuffle(x)
        >>> print(output.shape)
        (4,)
    """
    random_shuffle_ = _get_cache_prim(RandomShuffle)(seed=seed, seed2=seed2)
    output = random_shuffle_(x)
    return output


def choice_with_mask(input_x, count=256, seed=0, seed2=0):
    """
    Generates a random sample as index tensor with a mask tensor from a given tensor.

    The input_x must be a tensor of rank not less than 1. If its rank is greater than or equal to 2,
    the first dimension specifies the number of samples.
    The index tensor and the mask tensor have the fixed shapes. The index tensor denotes the index of the nonzero
    sample, while the mask tensor denotes which elements in the index tensor are valid.

    Args:
        input_x (Tensor): The input tensor.
            The input tensor rank must be greater than or equal to 1 and less than or equal to 5.
        count (int): Number of items expected to get and the number must be greater than 0. Default: 256.
        seed (int): Random seed. Default: 0.
        seed2 (int): Random seed2. Default: 0.

    Returns:
        Two tensors, the first one is the index tensor and the other one is the mask tensor.

        - **index** (Tensor) - The output shape is 2-D.
        - **mask** (Tensor) - The output shape is 1-D.

    Raises:
        TypeError: If `count` is not an int.
        TypeError: If neither `seed` nor `seed2` is an int.
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.ones(shape=[240000, 4]).astype(np.bool))
        >>> output_y, output_mask = ops.choice_with_mask(input_x)
        >>> result = output_y.shape
        >>> print(result)
        (256, 2)
        >>> result = output_mask.shape
        >>> print(result)
        (256,)
    """
    choice_with_mask_ = _get_cache_prim(RandomChoiceWithMask)(count=count, seed=seed, seed2=seed2)
    output = choice_with_mask_(input_x)
    return output


__all__ = [
    'standard_laplace',
    'random_categorical',
    'uniform',
    'standard_normal',
    'random_gamma',
    'uniform_candidate_sampler',
    'random_poisson',
    'random_shuffle',
    'choice_with_mask'
]
__all__.sort()
