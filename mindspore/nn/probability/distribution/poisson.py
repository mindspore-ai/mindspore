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
"""Poisson Distribution"""
import numpy as np
from mindspore.ops import operations as P
from mindspore.ops import composite as C
import mindspore.nn as nn
from mindspore._checkparam import Validator
from mindspore.common import dtype as mstype
from .distribution import Distribution
from ._utils.utils import check_greater_zero
from ._utils.custom_ops import exp_generic, log_generic


class Poisson(Distribution):
    """
    Poisson Distribution.

    Args:
        rate (list, numpy.ndarray, Tensor): The rate of the Poisson distribution..
        seed (int): The seed used in sampling. The global seed is used if it is None. Default: None.
        dtype (mindspore.dtype): The type of the event samples. Default: mstype.float32.
        name (str): The name of the distribution. Default: 'Poisson'.

    Supported Platforms:
        ``Ascend``

    Note:
        `rate` must be strictly greater than 0.
        `dist_spec_args` is `rate`.

    Examples:
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> import mindspore.nn.probability.distribution as msd
        >>> from mindspore import Tensor
        >>> # To initialize an Poisson distribution of the rate 0.5.
        >>> p1 = msd.Poisson([0.5], dtype=mindspore.float32)
        >>> # An Poisson distribution can be initialized without arguments.
        >>> # In this case, `rate` must be passed in through `args` during function calls.
        >>> p2 = msd.Poisson(dtype=mindspore.float32)
        >>>
        >>> # Here are some tensors used below for testing
        >>> value = Tensor([1, 2, 3], dtype=mindspore.int32)
        >>> rate_a = Tensor([0.6], dtype=mindspore.float32)
        >>> rate_b = Tensor([0.2, 0.5, 0.4], dtype=mindspore.float32)
        >>>
        >>> # Private interfaces of probability functions corresponding to public interfaces, including
        >>> # `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, and `log_survival`, are the same as follows.
        >>> # Args:
        >>> #     value (Tensor): the value to be evaluated.
        >>> #     rate (Tensor): the rate of the distribution. Default: self.rate.
        >>> # Examples of `prob`.
        >>> # Similar calls can be made to other probability functions
        >>> # by replacing `prob` by the name of the function.
        >>> ans = p1.prob(value)
        >>> print(ans.shape)
        (3,)
        >>> # Evaluate with respect to distribution b.
        >>> ans = p1.prob(value, rate_b)
        >>> print(ans.shape)
        (3,)
        >>> # `rate` must be passed in during function calls.
        >>> ans = p2.prob(value, rate_a)
        >>> print(ans.shape)
        (3,)
        >>> # Functions `mean`, `mode`, `sd`, and 'var' have the same arguments as follows.
        >>> # Args:
        >>> #     rate (Tensor): the rate of the distribution. Default: self.rate.
        >>> # Examples of `mean`, `sd`, `mode`, and `var` are similar.
        >>> ans = p1.mean() # return 2
        >>> print(ans.shape)
        (1,)
        >>> ans = p1.mean(rate_b) # return 1 / rate_b
        >>> print(ans.shape)
        (3,)
        >>> # `rate` must be passed in during function calls.
        >>> ans = p2.mean(rate_a)
        >>> print(ans.shape)
        (1,)
        >>> # Examples of `sample`.
        >>> # Args:
        >>> #     shape (tuple): the shape of the sample. Default: ()
        >>> #     probs1 (Tensor): the rate of the distribution. Default: self.rate.
        >>> ans = p1.sample()
        >>> print(ans.shape)
        (1, )
        >>> ans = p1.sample((2,3))
        >>> print(ans.shape)
        (2, 3, 1)
        >>> ans = p1.sample((2,3), rate_b)
        >>> print(ans.shape)
        (2, 3, 3)
        >>> ans = p2.sample((2,3), rate_a)
        >>> print(ans.shape)
        (2, 3, 1)
    """

    def __init__(self,
                 rate=None,
                 seed=None,
                 dtype=mstype.float32,
                 name="Poisson"):
        """
        Constructor of Poisson.
        """
        param = dict(locals())
        param['param_dict'] = {'rate': rate}
        valid_dtype = mstype.int_type + mstype.uint_type + mstype.float_type
        Validator.check_type_name("dtype", dtype, valid_dtype, type(self).__name__)

        # As some operators can't accept scalar input, check the type here
        if isinstance(rate, (int, float)):
            raise TypeError("Input rate can't be scalar")

        super(Poisson, self).__init__(seed, dtype, name, param)

        self._rate = self._add_parameter(rate, 'rate')
        if self.rate is not None:
            check_greater_zero(self.rate, 'rate')

        # ops needed for the class
        self.exp = exp_generic
        self.log = log_generic
        self.squeeze = P.Squeeze(0)
        self.cast = P.Cast()
        self.floor = P.Floor()
        self.dtypeop = P.DType()
        self.shape = P.Shape()
        self.fill = P.Fill()
        self.less = P.Less()
        self.equal = P.Equal()
        self.select = P.Select()
        self.lgamma = nn.LGamma()
        self.igamma = nn.IGamma()
        self.poisson = C.poisson

    def extend_repr(self):
        if self.is_scalar_batch:
            s = f'rate = {self.rate}'
        else:
            s = f'batch_shape = {self._broadcast_shape}'
        return s

    @property
    def rate(self):
        """
        Return `rate` of the distribution after casting to dtype.
        """
        return self._rate

    def _get_dist_type(self):
        return "Poisson"

    def _get_dist_args(self, rate=None):
        if rate is not None:
            self.checktensor(rate, 'rate')
        else:
            rate = self.rate
        return (rate,)

    def _mean(self, rate=None):
        r"""
        .. math::
            MEAN(POISSON) = \lambda.
        """
        rate = self._check_param_type(rate)
        return rate

    def _mode(self, rate=None):
        r"""
        .. math::
            MODE(POISSON) = \lfloor{\lambda}.
        """
        rate = self._check_param_type(rate)
        return self.floor(rate)

    def _var(self, rate=None):
        r"""
        .. math::
            VAR(POISSON) = \lambda.
        """
        rate = self._check_param_type(rate)
        return rate

    def _log_prob(self, value, rate=None):
        r"""
        Log probability density function of Poisson distributions.

        Args:
            Args:
            value (Tensor): The value to be evaluated.
            rate (Tensor): The rate of the distribution. Default: self.rate.

        Note:
            `value` must be greater or equal to zero.

        .. math::
            log_pdf(x) = x * \log(\lambda) - \lambda - \log(\Gamma(x)) if x >= 0 else -inf
        """
        value = self._check_value(value, "value")
        value = self.cast(value, self.dtype)
        rate = self._check_param_type(rate)
        log_rate = self.log(rate)
        zeros = self.fill(self.dtypeop(value), self.shape(value), 0.0)
        inf = self.fill(self.dtypeop(value), self.shape(value), np.inf)
        safe_x = self.select(self.less(value, zeros), zeros, value)
        y = log_rate * safe_x - self.lgamma(safe_x + 1.)
        comp = self.equal(value, safe_x)
        log_unnormalized_prob = self.select(comp, y, -inf)
        log_normalization = self.exp(log_rate)
        return log_unnormalized_prob - log_normalization

    def _cdf(self, value, rate=None):
        r"""
        Cumulative distribution function (cdf) of Poisson distributions.

        Args:
            value (Tensor): The value to be evaluated.
            rate (Tensor): The rate of the distribution. Default: self.rate.

        Note:
            `value` must be greater or equal to zero.

        .. math::
            cdf(x) = \Gamma(x + 1) if x >= 0 else 0
        """
        value = self._check_value(value, 'value')
        value = self.cast(value, self.dtype)
        rate = self._check_param_type(rate)
        zeros = self.fill(self.dtypeop(value), self.shape(value), 0.0)
        comp = self.less(value, zeros)
        safe_x = self.select(comp, zeros, value)
        cdf = 1. - self.igamma(1. + safe_x, rate)
        return self.select(comp, zeros, cdf)

    def _sample(self, shape=(), rate=None):
        """
        Sampling.

        Args:
            shape (tuple): The shape of the sample. Default: ().
            rate (Tensor): The rate of the distribution. Default: self.rate.

        Returns:
            Tensor, shape is shape + batch_shape.
        """
        shape = self.checktuple(shape, 'shape')
        rate = self._check_param_type(rate)

        # now Poisson sampler supports only fp32
        rate = self.cast(rate, mstype.float32)
        origin_shape = shape + self.shape(rate)
        if origin_shape == ():
            sample_shape = (1,)
        else:
            sample_shape = origin_shape
        sample_poisson = self.poisson(sample_shape, rate, self.seed)
        value = self.cast(sample_poisson, self.dtype)
        if origin_shape == ():
            value = self.squeeze(value)
        return value
