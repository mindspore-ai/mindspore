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
"""StudentT Distribution"""
from __future__ import absolute_import
from __future__ import division
import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore._checkparam import Validator
from mindspore.common import dtype as mstype
from mindspore.nn.probability.distribution import Distribution
from mindspore.nn.probability.distribution._utils.utils import check_greater_zero


class StudentT(Distribution):
    r"""
    StudentT distribution.
    A StudentT distribution is a continuous distribution with the range :math:`(-\inf, \inf)`
    and the probability density function:

    .. math::
        f(x, \nu, \mu, \sigma) = (1 + y^2 / \nu)^{(-0.5*(\nu + 1))} / Z

    where :math:`y = (x-\mu)/\sigma`,
    :math:`Z = abs(\sigma) * \sqrt{(\nu * \pi)} * \Gamma(0.5 * \nu) / \Gamma(0.5 * (\nu + 1))`,
    :math:`\nu, \mu, \sigma` are the degrees of freedom , mean and scale of the laplace distribution respectively.

    Args:
        df (int, float, list, numpy.ndarray, Tensor): The degrees of freedom. Default: None.
        mean (int, float, list, numpy.ndarray, Tensor): The mean of the distribution. Default: None.
        sd (int, float, list, numpy.ndarray, Tensor): The standard deviation of the distribution. Default: None.
        seed (int): The seed used in sampling. The global seed is used if it is None. Default: None.
        dtype (mindspore.dtype): The type of the event samples. Default: mstype.float32.
        name (str): The name of the distribution. Default: 'StudentT'.

    Note:
        - `df` must be greater than zero.
        - `sd` must be greater than zero.
        - `dist_spec_args` are `mean` and `sd`.
        - `dtype` must be a float type because StudentT distributions are continuous.

    Raises:
        ValueError: When df <= 0.
        ValueError: When sd <= 0.
        TypeError: When the input `dtype` is not a subclass of float.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> import mindspore.nn.probability.distribution as msd
        >>> from mindspore import Tensor
        >>> # To initialize a StudentT distribution of the df 2.0, the mean 3.0 and the standard deviation 4.0.
        >>> n1 = msd.StudentT(2.0, 3.0, 4.0, dtype=mindspore.float32)
        >>> # A StudentT distribution can be initialized without arguments.
        >>> # In this case, `df`, `mean` and `sd` must be passed in through arguments.
        >>> n2 = msd.StudentT(dtype=mindspore.float32)
        >>> # Here are some tensors used below for testing
        >>> value = Tensor([1.0, 2.0, 3.0], dtype=mindspore.float32)
        >>> df_a = Tensor([2.0], dtype=mindspore.float32)
        >>> mean_a = Tensor([2.0], dtype=mindspore.float32)
        >>> sd_a = Tensor([2.0, 2.0, 2.0], dtype=mindspore.float32)
        >>> df_b = Tensor([1.0], dtype=mindspore.float32)
        >>> mean_b = Tensor([1.0], dtype=mindspore.float32)
        >>> sd_b = Tensor([1.0, 1.5, 2.0], dtype=mindspore.float32)
        >>> ans = n1.log_prob(value)
        >>> print(ans.shape)
        (3,)
        >>> # Evaluate with respect to the distribution b.
        >>> ans = n1.log_prob(value, df_b, mean_b, sd_b)
        >>> print(ans.shape)
        (3,)
        >>> # `mean` and `sd` must be passed in during function calls
        >>> ans = n2.log_prob(value, df_a, mean_a, sd_a)
        >>> print(ans.shape)
        (3,)
    """

    def __init__(self,
                 df=None,
                 mean=None,
                 sd=None,
                 seed=None,
                 dtype=mstype.float32,
                 name="StudentT"):
        """
        Constructor of StudentT.
        """
        param = dict(locals())
        param['param_dict'] = {'df': df, 'mean': mean, 'sd': sd}
        valid_dtype = mstype.float_type
        Validator.check_type_name("dtype", dtype, valid_dtype, type(self).__name__)
        super(StudentT, self).__init__(seed, dtype, name, param)

        self._df_value = self._add_parameter(df, 'df')
        self._mean_value = self._add_parameter(mean, 'mean')
        self._sd_value = self._add_parameter(sd, 'sd')
        if self._sd_value is not None:
            check_greater_zero(self._sd_value, "Standard deviation")
        if self._df_value is not None:
            check_greater_zero(self._df_value, "Degrees of freedom")
        self.log1p = P.Log1p()
        self.log = P.Log()
        self.cast = P.Cast()
        self.abs = P.Abs()
        self.half = 0.5
        self.half_log_pi = 0.5 * np.log(np.pi)
        self.lgamma = nn.LGamma()

    def _log_prob(self, value, df=None, mean=None, sd=None):
        r"""
        Evaluate log probability.

        Args:
            value (Tensor): The value to be evaluated.
            df (Tensor): The degrees of freedom of the distribution. Default: self._df_value.
            mean (Tensor): The mean of the distribution. Default: self._mean_value.
            sd (Tensor): The standard deviation the distribution. Default: self._sd_value.

        .. math::
            L(x) = -0.5 * (\nu + 1.) * \log((x - \mu) / \sigma + 1.)) + \log(\sqrt(\pi * \mu * \sigma^2))
                + log(\Gamma(\nu / 2.)) - log(\Gamma((\nu + 1.) / 2.))
        """
        value = self._check_value(value, 'value')
        value = self.cast(value, self.dtype)
        df, mean, sd = self._check_param_type(df, mean, sd)

        y = (value - mean) / sd
        log_unnormalized_prob = -0.5 * (df + 1.) * self.log1p(y**2. / df)
        log_normalization = self.log(self.abs(sd)) + 0.5 * self.log(df) + self.half_log_pi + \
                            self.lgamma(self.half * df) - self.lgamma(self.half * (df + 1.))
        return log_unnormalized_prob - log_normalization
