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

"""Operations for random number generatos."""

from .. import operations as P
from .. import functional as F
from ..primitive import constexpr
from .multitype_ops import _constexpr_utils as const_utils
from ...common import dtype as mstype
from ...common.tensor import Tensor
from ..._checkparam import Validator as validator
from ..._checkparam import check_int_positive
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
        seed(Int): the graph-level seed value that to be set.

    Examples:
        >>> C.set_seed(10)
    """
    check_int_positive(seed)
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
        >>> C.get_seed(10)
    """
    return _GRAPH_SEED


def normal(shape, mean, stddev, seed=0):
    """
    Generates random numbers according to the Normal (or Gaussian) random number distribution.
    It is defined as:

    Args:
        shape (tuple): The shape of random tensor to be generated.
        mean (Tensor): The mean μ distribution parameter, which specifies the location of the peak.
          With float32 data type.
        stddev (Tensor): The deviation σ distribution parameter. With float32 data type.
        seed (int): Seed is used as entropy source for Random number engines generating pseudo-random numbers.
          Default: 0.

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
    seed1 = get_seed()
    seed2 = seed
    stdnormal = P.StandardNormal(seed1, seed2)
    rnd = stdnormal(shape)
    value = rnd * stddev + mean
    return value


def multinomial(inputs, num_sample=None, replacement=True, seed=0):
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
        - **num_samples** (int) - number of samples to draw, default None.
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
        a = Tensor(0.0, mstype.float32)
        b = Tensor(1.0, mstype.float32)
        uniform = P.UniformReal(seed=seed)((n_dist * num_sample,), a, b)
        if n_dist != 1:
            uniform = reshape(uniform, (n_dist, num_sample))
        vals = P.RealDiv()(P.Log()(uniform), inputs + 1e-6)
        _, indices = P.TopK()(vals, num_sample)
        return indices
    return P.Multinomial(seed=seed)(inputs, num_sample)
