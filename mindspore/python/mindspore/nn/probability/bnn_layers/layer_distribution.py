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
"""Initialize normal distributions"""
import numpy as np
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
from ...cell import Cell
from ..distribution.normal import Normal

__all__ = ['NormalPrior', 'NormalPosterior']


class NormalPrior(Cell):
    r"""
    To initialize a normal distribution of mean 0 and standard deviation 0.1.

    Args:
        dtype (mindspore.dtype): The argument is used to define the data type of the output tensor.
            Default: ``mindspore.float32`` .
        mean (int, float): Mean of normal distribution. Default: ``0`` .
        std (int, float): Standard deviation of normal distribution. Default: ``0.1`` .

    Returns:
        Cell, a normal distribution.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self, dtype=mstype.float32, mean=0, std=0.1):
        super(NormalPrior, self).__init__()
        self.normal = Normal(mean, std, dtype=dtype)

    def construct(self, *inputs):
        return self.normal(*inputs)


class NormalPosterior(Cell):
    r"""
    Build Normal distributions with trainable parameters.

    Args:
        name (str): Name prepended to trainable parameter.
        shape (list, tuple): Shape of the mean and standard deviation.
        dtype (mindspore.dtype): The argument is used to define the data type of the output tensor.
            Default: ``mindspore.float32`` .
        loc_mean (int, float): Mean of distribution to initialize trainable parameters. Default: ``0`` .
        loc_std (int, float): Standard deviation of distribution to initialize trainable parameters. Default: ``0.1`` .
        untransformed_scale_mean (int, float): Mean of distribution to initialize trainable parameters.
            Default: ``-5`` .
        untransformed_scale_std (int, float): Standard deviation of distribution to initialize trainable parameters.
            Default: ``0.1`` .

    Returns:
        Cell, a normal distribution.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self,
                 name,
                 shape,
                 dtype=mstype.float32,
                 loc_mean=0,
                 loc_std=0.1,
                 untransformed_scale_mean=-5,
                 untransformed_scale_std=0.1):
        super(NormalPosterior, self).__init__()
        if not isinstance(name, str):
            raise TypeError('The type of `name` must be `str`')

        if not isinstance(shape, (tuple, list)):
            raise TypeError('The type of `shape` must be `tuple` or `list`')

        if isinstance(loc_mean, bool) or not isinstance(loc_mean, (int, float)):
            raise TypeError('The type of `loc_mean` must be `int` or `float`')

        if isinstance(untransformed_scale_mean, bool) or not isinstance(untransformed_scale_mean, (int, float)):
            raise TypeError('The type of `untransformed_scale_mean` must be `int` or `float`')

        if isinstance(loc_std, bool) or not (isinstance(loc_std, (int, float)) and loc_std >= 0):
            raise TypeError('The type of `loc_std` must be `int` or `float` and its value must > 0')

        if isinstance(loc_std, bool) or not (isinstance(untransformed_scale_std, (int, float)) and
                                             untransformed_scale_std >= 0):
            raise TypeError('The type of `untransformed_scale_std` must be `int` or `float` and '
                            'its value must > 0')

        self.mean = Parameter(
            Tensor(np.random.normal(loc_mean, loc_std, shape), dtype=dtype), name=name + '_mean')

        self.untransformed_std = Parameter(
            Tensor(np.random.normal(untransformed_scale_mean, untransformed_scale_std, shape), dtype=dtype),
            name=name + '_untransformed_std')

        self.normal = Normal()

    def _std_trans(self, std_pre):
        """Transform std_pre to prevent its value being zero."""
        std = 1e-6 + P.Log()(P.Exp()(std_pre) + 1)
        return std

    def construct(self, *inputs):
        std = self._std_trans(self.untransformed_std)
        return self.normal(*inputs, mean=self.mean, sd=std)


def normal_post_fn(name, shape):
    """Provide normal posterior distribution."""
    return NormalPosterior(name=name, shape=shape)
