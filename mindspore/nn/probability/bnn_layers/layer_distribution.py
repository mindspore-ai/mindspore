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
        dtype (class `mindspore.dtype`): The argument is used to define the data type of the output tensor.
            Default: mindspore.float32.
        mean (int, float): Mean of normal distribution.
        std (int, float): Standard deviation of normal distribution.

    Returns:
        Cell, a normal distribution.
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
        shape (list): Shape of the mean and standard deviation.
        dtype (class `mindspore.dtype`): The argument is used to define the data type of the output tensor.
            Default: mindspore.float32.
        loc_mean ( float, array_like of floats): Mean of distribution to initialize trainable parameters. Default: 0.
        loc_std ( float, array_like of floats): Standard deviation of distribution to initialize trainable parameters.
            Default: 0.1.
        untransformed_scale_mean ( float, array_like of floats): Mean of distribution to initialize trainable
            parameters. Default: -5.
        untransformed_scale_std ( float, array_like of floats): Standard deviation of distribution to initialize
            trainable parameters. Default: 0.1.

    Returns:
        Cell, a normal distribution.
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
            raise ValueError('The type of `name` should be `str`')

        self.mean = Parameter(
            Tensor(np.random.normal(loc_mean, loc_std, shape), dtype=dtype), name=name + '_mean')

        self.untransformed_std = Parameter(
            Tensor(np.random.normal(untransformed_scale_mean, untransformed_scale_std, shape), dtype=dtype),
            name=name + '_untransformed_std')

        self.normal = Normal()

    def std_trans(self, std_pre):
        """Transform std_pre to prevent its value being zero."""
        std = 1e-6 + P.Log()(P.Exp()(std_pre) + 1)
        return std

    def construct(self, *inputs):
        std = self.std_trans(self.untransformed_std)
        return self.normal(*inputs, mean=self.mean, sd=std)
