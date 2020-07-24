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

# set graph-level RNG seed
_GRAPH_SEED = 0

@constexpr
def set_seed(seed):
    global _GRAPH_SEED
    _GRAPH_SEED = seed

@constexpr
def get_seed():
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
        >>> C.set_seed(10)
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
