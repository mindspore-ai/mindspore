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
"""Laplace Distribution"""
from __future__ import absolute_import
from __future__ import division
from mindspore.ops import operations as P
from mindspore._checkparam import Validator
from mindspore.common import dtype as mstype
from mindspore.nn.probability.distribution import Distribution
from mindspore.nn.probability.distribution._utils.utils import check_greater_zero


class Laplace(Distribution):
    r"""
    Laplace distribution.
    A Laplace distribution is a continuous distribution with the range :math:`(-\inf, \inf)`
    and the probability density function:

    .. math::
        f(x, \mu, b) = 1 / (2. * b) * \exp(-abs(x - \mu) / b).

    where :math:`\mu, b` are the mean and the scale of the laplace distribution respectively.

    Args:
        mean (int, float, list, numpy.ndarray, Tensor): The mean of the distribution. Default: None.
        sd (int, float, list, numpy.ndarray, Tensor): The standard deviation of the distribution. Default: None.
        seed (int): The seed used in sampling. The global seed is used if it is None. Default: None.
        dtype (mindspore.dtype): The type of the event samples. Default: mstype.float32.
        name (str): The name of the distribution. Default: 'Laplace'.

    Note:
        - `sd` must be greater than zero.
        - `dist_spec_args` are `mean` and `sd`.
        - `dtype` must be a float type because Laplace distributions are continuous.

    Raises:
        ValueError: When sd <= 0.
        TypeError: When the input `dtype` is not a subclass of float.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> from mindspore.nn.probability.distribution import Laplace
        >>> from mindspore import Tensor
        >>> # To initialize a Laplace distribution of the mean 3.0 and the standard deviation 4.0.
        >>> n1 = Laplace(3.0, 4.0, dtype=mindspore.float32)
        >>> # A Laplace distribution can be initialized without arguments.
        >>> # In this case, `mean` and `sd` must be passed in through arguments.
        >>> n2 = Laplace(dtype=mindspore.float32)
        >>> # Here are some tensors used below for testing
        >>> value = Tensor([1.0, 2.0, 3.0], dtype=mindspore.float32)
        >>> mean_a = Tensor([2.0], dtype=mindspore.float32)
        >>> sd_a = Tensor([2.0, 2.0, 2.0], dtype=mindspore.float32)
        >>> mean_b = Tensor([1.0], dtype=mindspore.float32)
        >>> sd_b = Tensor([1.0, 1.5, 2.0], dtype=mindspore.float32)
        >>> ans = n1.log_prob(value)
        >>> print(ans.shape)
        (3,)
        >>> # Evaluate with respect to the distribution b.
        >>> ans = n1.log_prob(value, mean_b, sd_b)
        >>> print(ans.shape)
        (3,)
        >>> # `mean` and `sd` must be passed in during function calls
        >>> ans = n2.log_prob(value, mean_a, sd_a)
        >>> print(ans.shape)
        (3,)
    """

    def __init__(self,
                 mean=None,
                 sd=None,
                 seed=None,
                 dtype=mstype.float32,
                 name="Laplace"):
        """
        Constructor of Laplace.
        """
        param = dict(locals())
        param['param_dict'] = {'mean': mean, 'sd': sd}
        valid_dtype = mstype.float_type
        Validator.check_type_name("dtype", dtype, valid_dtype, type(self).__name__)
        super(Laplace, self).__init__(seed, dtype, name, param)

        self._mean_value = self._add_parameter(mean, 'mean')
        self._sd_value = self._add_parameter(sd, 'sd')
        if self._sd_value is not None:
            check_greater_zero(self._sd_value, "Standard deviation")

        self.log = P.Log()
        self.cast = P.Cast()
        self.abs = P.Abs()

    def _log_prob(self, value, mean=None, sd=None):
        r"""
        Evaluate log probability.

        Args:
            value (Tensor): The value to be evaluated.
            mean (Tensor): The mean of the distribution. Default: self._mean_value.
            sd (Tensor): The standard deviation the distribution. Default: self._sd_value.

        .. math::
            L(x) = -1* \abs{\frac{x - \mu}{\sigma}} - \log(2. * \sigma))
        """
        value = self._check_value(value, 'value')
        value = self.cast(value, self.dtype)
        mean, sd = self._check_param_type(mean, sd)

        pdf = -1.0 * (self.abs((value - mean) / sd)) - self.log(2. * sd)
        return pdf
