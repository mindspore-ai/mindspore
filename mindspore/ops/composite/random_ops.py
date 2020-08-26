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

"""Operations for random number generators."""

from .. import operations as P
from .. import functional as F
from ..primitive import constexpr
from .multitype_ops import _constexpr_utils as const_utils
from ...common import dtype as mstype
from ..._checkparam import Validator as validator
from ..._checkparam import Rel

# set graph-level RNG seed
_GRAPH_SEED = 0

@constexpr
def set_seed(seed):
    """
    Set the graph-level seed.
    Graph-level seed is used as a global variable, that can be used in different ops in case op-level seed is not set.
    If op-level seed is 0, use graph-level seed; if op-level seed is also 0, the system would generate a
    random seed.

    Args:
        seed(Int): the graph-level seed value that to be set. Must be non-negative.

    Examples:
        >>> C.set_seed(10)
    """
    const_utils.check_non_negative("seed", seed, "set_seed")
    global _GRAPH_SEED
    _GRAPH_SEED = seed

@constexpr
def get_seed():
    """
    Get the graph-level seed.
    Graph-level seed is used as a global variable, that can be used in different ops in case op-level seed is not set.
    If op-level seed is 0, use graph-level seed; if op-level seed is also 0, the system would generate a
    random seed.

    Returns:
        Interger. The current graph-level seed.

    Examples:
        >>> C.get_seed()
    """
    return _GRAPH_SEED

def normal(shape, mean, stddev, seed=0):
    """
    Generates random numbers according to the Normal (or Gaussian) random number distribution.

    Args:
        shape (tuple): The shape of random tensor to be generated.
        mean (Tensor): The mean μ distribution parameter, which specifies the location of the peak.
          With float32 data type.
        stddev (Tensor): The deviation σ distribution parameter. With float32 data type.
        seed (int): Seed is used as entropy source for Random number engines generating pseudo-random numbers.
          Must be non-negative. Default: 0.

    Returns:
        Tensor. The shape should be the broadcasted shape of Input "shape" and shapes of mean and stddev.
        The dtype is float32.

    Examples:
        >>> shape = (4, 16)
        >>> mean = Tensor(1.0, mstype.float32)
        >>> stddev = Tensor(1.0, mstype.float32)
        >>> output = C.normal(shape, mean, stddev, seed=5)
    """
    mean_dtype = F.dtype(mean)
    stddev_dtype = F.dtype(stddev)
    const_utils.check_tensors_dtype_same(mean_dtype, mstype.float32, "normal")
    const_utils.check_tensors_dtype_same(stddev_dtype, mstype.float32, "normal")
    const_utils.check_non_negative("seed", seed, "normal")
    seed1 = get_seed()
    seed2 = seed
    stdnormal = P.StandardNormal(seed1, seed2)
    random_normal = stdnormal(shape)
    value = random_normal * stddev + mean
    return value

def uniform(shape, minval, maxval, seed=0, dtype=mstype.float32):
    """
    Generates random numbers according to the Uniform random number distribution.

    Note:
        The number in tensor minval should be strictly less than maxval at any position after broadcasting.

    Args:
        shape (tuple): The shape of random tensor to be generated.
        minval (Tensor): The a distribution parameter.
          It defines the minimum possibly generated value. With int32 or float32 data type.
          If dtype is int32, only one number is allowed.
        maxval (Tensor): The b distribution parameter.
          It defines the maximum possibly generated value. With int32 or float32 data type.
          If dtype is int32, only one number is allowed.
        seed (int): Seed is used as entropy source for Random number engines generating pseudo-random numbers.
          Must be non-negative. Default: 0.

    Returns:
        Tensor. The shape should be the broadcasted shape of Input "shape" and shapes of minval and maxval.
        The dtype is designated as the input `dtype`.

    Examples:
        >>> For discrete uniform distribution, only one number is allowed for both minval and maxval:
        >>> shape = (4, 2)
        >>> minval = Tensor(1, mstype.int32)
        >>> maxval = Tensor(2, mstype.int32)
        >>> output = C.uniform(shape, minval, maxval, seed=5)
        >>>
        >>> For continuous uniform distribution, minval and maxval can be multi-dimentional:
        >>> shape = (4, 2)
        >>> minval = Tensor([1.0, 2.0], mstype.float32)
        >>> maxval = Tensor([4.0, 5.0], mstype.float32)
        >>> output = C.uniform(shape, minval, maxval, seed=5)
    """
    minval_dtype = F.dtype(minval)
    maxval_dtype = F.dtype(maxval)
    const_utils.check_tensors_dtype_same(minval_dtype, dtype, "uniform")
    const_utils.check_tensors_dtype_same(maxval_dtype, dtype, "uniform")
    const_utils.check_non_negative("seed", seed, "uniform")
    seed1 = get_seed()
    seed2 = seed
    if const_utils.is_same_type(dtype, mstype.int32):
        random_uniform = P.UniformInt(seed1, seed2)
        value = random_uniform(shape, minval, maxval)
    else:
        uniform_real = P.UniformReal(seed1, seed2)
        random_uniform = uniform_real(shape)
        value = random_uniform * (maxval - minval) + minval
    return value

def gamma(shape, alpha, beta, seed=0):
    """
    Generates random numbers according to the Gamma random number distribution.

    Args:
        shape (tuple): The shape of random tensor to be generated.
        alpha (Tensor): The alpha α distribution parameter. With float32 data type.
        beta (Tensor): The beta β distribution parameter. With float32 data type.
        seed (int): Seed is used as entropy source for Random number engines generating pseudo-random numbers.
          Must be non-negative. Default: 0.

    Returns:
        Tensor. The shape should be the broadcasted shape of Input "shape" and shapes of alpha and beta.
        The dtype is float32.

    Examples:
        >>> shape = (4, 16)
        >>> alpha = Tensor(1.0, mstype.float32)
        >>> beta = Tensor(1.0, mstype.float32)
        >>> output = C.gamma(shape, alpha, beta, seed=5)
    """
    const_utils.check_non_negative("seed", seed, "gamma")
    seed1 = get_seed()
    seed2 = seed
    random_gamma = P.Gamma(seed1, seed2)
    value = random_gamma(shape, alpha, beta)
    return value

def poisson(shape, mean, seed=0):
    """
    Generates random numbers according to the Poisson random number distribution.

    Args:
        shape (tuple): The shape of random tensor to be generated.
        mean (Tensor): The mean μ distribution parameter. With float32 data type.
        seed (int): Seed is used as entropy source for Random number engines generating pseudo-random numbers.
          Must be non-negative. Default: 0.

    Returns:
        Tensor. The shape should be the broadcasted shape of Input "shape" and shapes of mean.
        The dtype is float32.

    Examples:
        >>> shape = (4, 16)
        >>> mean = Tensor(1.0, mstype.float32)
        >>> output = C.poisson(shape, mean, seed=5)
    """
    const_utils.check_non_negative("seed", seed, "poisson")
    seed1 = get_seed()
    seed2 = seed
    random_poisson = P.Poisson(seed1, seed2)
    value = random_poisson(shape, mean)
    return value

def multinomial(inputs, num_sample, replacement=True, seed=0):
    r"""
    Returns a tensor sampled from the multinomial probability distribution located in the corresponding
    row of tensor input.

    Note:
        The rows of input do not need to sum to one (in which case we use the values as weights),
        but must be non-negative, finite and have a non-zero sum.
    Args:
        seed (int): Seed data is used as entropy source for Random number engines generating pseudo-random numbers.
          Default: 0.

    Inputs:
        - **input** (Tensor) - the input tensor containing probabilities, must be 1 or 2 dims.
        - **num_samples** (int) - number of samples to draw.
        - **replacement** (bool, optional) - whether to draw with replacement or not, default True.

    Outputs:
        Tensor. have the same rows with input, each row has num_samples sampled indices.

    Examples:
        >>> input = Tensor([0, 9, 4, 0], mstype.float32)
        >>> output = C.multinomial(input, 2, True)
    """
    shape = P.Shape()
    reshape = P.Reshape()
    validator.check_value_type('replacement', replacement, (bool,), None)
    validator.check_value_type('num_sample', num_sample, (int,), None)
    validator.check_integer("num_sample", num_sample, 0, Rel.GT, None)
    if inputs.dim() != 1 and inputs.dim() != 2:
        raise ValueError("inputs dim must be 1d or 2d")
    if not replacement:
        if shape(inputs)[-1] < num_sample:
            raise ValueError("num_sample must be less than shape(input)[-1] without replacement")
        n_dist = 1
        if len(shape(inputs)) > 1:
            n_dist = shape(inputs)[-2]
        random_uniform = P.UniformReal(seed=seed)((n_dist * num_sample,))
        if n_dist != 1:
            random_uniform = reshape(random_uniform, (n_dist, num_sample))
        vals = P.RealDiv()(P.Log()(random_uniform), inputs + 1e-6)
        _, indices = P.TopK()(vals, num_sample)
        return indices
    return P.Multinomial(seed=seed)(inputs, num_sample)
