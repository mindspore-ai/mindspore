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
"""HalfNormal Distribution"""
from __future__ import absolute_import
from __future__ import division
import numpy as np
from mindspore import ops
from mindspore.ops import operations as P
from mindspore import _checkparam as Validator
from mindspore.common import dtype as mstype
from mindspore.nn.probability.distribution import Distribution
from mindspore.nn.probability.distribution._utils.utils import check_greater_zero


class HalfNormal(Distribution):
    r"""
    HalfNormal distribution.
    A HalfNormal distribution is a continuous distribution with the range :math:`[\mu, \inf)`
    and the probability density function:

    .. math::
        f(x, \mu, \sigma) = 1 / \sigma\sqrt{2\pi} \exp(-(x - \mu)^2 / 2\sigma^2).

    where :math:`\mu, \sigma` are the mean and the standard deviation of the half normal distribution respectively.

    Args:
        mean (Union[int, float, list, numpy.ndarray, Tensor], optional): The mean of the distribution.
            If this arg is ``None`` , then the mean of the distribution will be passed in runtime. Default: ``None`` .
        sd (Union[int, float, list, numpy.ndarray, Tensor], optional): The standard deviation of the distribution.
            If this arg is ``None`` , then the sd of the distribution will be passed in runtime. Default: ``None`` .
        seed (int, optional): The seed used in sampling. The global seed is used if it is None. Default: ``None`` .
        dtype (mindspore.dtype, optional): The type of the event samples. Default: ``mstype.float32`` .
        name (str, optional): The name of the distribution. Default: ``'HalfNormal'`` .

    Note:
        - `sd` must be greater than zero.
        - `dtype` must be a float type because HalfNormal distributions are continuous.
        - If the arg `mean` or `sd` is passed in runtime, then it will be used as the parameter value.
          Otherwise, the value passed in the constructor will be used.

    Raises:
        ValueError: When sd <= 0.
        TypeError: When the input `dtype` is not a subclass of float.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> from mindspore.nn.probability.distribution import HalfNormal
        >>> from mindspore import Tensor
        >>> # To initialize a HalfNormal distribution of the mean 3.0 and the standard deviation 4.0.
        >>> n1 = HalfNormal(3.0, 4.0, dtype=mindspore.float32)
        >>> # A HalfNormal distribution can be initialized without arguments.
        >>> # In this case, `mean` and `sd` must be passed in through arguments.
        >>> hn = HalfNormal(dtype=mindspore.float32)
        >>> # Here are some tensors used below for testing
        >>> value = Tensor([1.0, 2.0, 3.0], dtype=mindspore.float32)
        >>> mean_a = Tensor([2.0], dtype=mindspore.float32)
        >>> sd_a = Tensor([2.0, 2.0, 2.0], dtype=mindspore.float32)
        >>> mean_b = Tensor([1.0], dtype=mindspore.float32)
        >>> sd_b = Tensor([1.0, 1.5, 2.5], dtype=mindspore.float32)
        >>> ans = n1.log_prob(value)
        >>> print(ans.shape)
        (3,)
        >>> # Evaluate with respect to the distribution b.
        >>> ans = n1.log_prob(value, mean_b, sd_b)
        >>> print(ans.shape)
        (3,)
        >>> # `mean` and `sd` must be passed in during function calls
        >>> ans = hn.log_prob(value, mean_a, sd_a)
        >>> print(ans.shape)
        (3,)
    """

    def __init__(self,
                 mean=None,
                 sd=None,
                 seed=None,
                 dtype=mstype.float32,
                 name="HalfNormal"):
        """
        Constructor of HalfNormal.
        """
        param = dict(locals())
        param['param_dict'] = {'mean': mean, 'sd': sd}
        valid_dtype = mstype.float_type
        Validator.check_type_name("dtype", dtype, valid_dtype, type(self).__name__)
        super(HalfNormal, self).__init__(seed, dtype, name, param)

        self._mean_value = self._add_parameter(mean, 'mean')
        self._sd_value = self._add_parameter(sd, 'sd')
        if self._sd_value is not None:
            check_greater_zero(self._sd_value, "Standard deviation")

        self.exp = P.Exp()
        self.cast = P.Cast()
        self.const = ops.scalar_to_tensor(np.sqrt(2. / np.pi))
        self.sq = P.Square()
        self.type = dtype

    def _prob(self, value, mean=None, sd=None):
        r"""
        Evaluate probability of the value of the HalfNormal distribution.

        Args:
            value (Tensor): The value to be evaluated.
            mean (Tensor, optional): The mean of the distribution. Default: self._mean_value.
            sd (Tensor, optional): The standard deviation the distribution. Default: self._sd_value.

        .. math::
            P(x) = 1 / \sigma \sqrt{2\pi} \exp(-(x - \mu)^2 / 2\sigma^2)
        """
        value = self._check_value(value, 'value')
        value = self.cast(value, self.dtype)
        mean, sd = self._check_param_type(mean, sd)

        coeff = self.const / sd
        pdf = coeff * self.exp(-0.5 * self.sq((value - mean) / sd))
        return pdf * self.cast(value >= 0, self.type)
