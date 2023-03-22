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
"""Gamma Distribution"""
import numpy as np
from mindspore.ops import operations as P
from mindspore.ops import composite as C
import mindspore.nn as nn
from mindspore._checkparam import Validator
from mindspore.common import dtype as mstype
from .distribution import Distribution
from ._utils.utils import check_greater_zero, check_distribution_name
from ._utils.custom_ops import log_generic


class Gamma(Distribution):
    r"""
    Gamma distribution.
    A Gamma distributio is a continuous distribution with the range :math:`(0, \inf)`
    and the probability density function:

    .. math::
        f(x, \alpha, \beta) = \beta^\alpha / \Gamma(\alpha) x^{\alpha - 1} \exp(-\beta x).

    where :math:`G` is the Gamma function,
    and :math:`\alpha, \beta` are the concentration and the rate of the distribution respectively.

    Args:
        concentration (int, float, list, numpy.ndarray, Tensor): The concentration,
          also know as alpha of the Gamma distribution. Default: None.
        rate (int, float, list, numpy.ndarray, Tensor): The rate, also know as
          beta of the Gamma distribution. Default: None.
        seed (int): The seed used in sampling. The global seed is used if it is None. Default: None.
        dtype (mindspore.dtype): The type of the event samples. Default: mstype.float32.
        name (str): The name of the distribution. Default: 'Gamma'.

    Note:
        `concentration` and `rate` must be greater than zero.
        `dist_spec_args` are `concentration` and `rate`.
        `dtype` must be a float type because Gamma distributions are continuous.

    Raises:
        ValueError: When concentration <= 0 or rate <= 0.
        TypeError: When the input `dtype` is not a subclass of float.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> import mindspore.nn.probability.distribution as msd
        >>> from mindspore import Tensor
        >>> # To initialize a Gamma distribution of the concentration 3.0 and the rate 4.0.
        >>> g1 = msd.Gamma([3.0], [4.0], dtype=mindspore.float32)
        >>> # A Gamma distribution can be initialized without arguments.
        >>> # In this case, `concentration` and `rate` must be passed in through arguments.
        >>> g2 = msd.Gamma(dtype=mindspore.float32)
        >>> # Here are some tensors used below for testing
        >>> value = Tensor([1.0, 2.0, 3.0], dtype=mindspore.float32)
        >>> concentration_a = Tensor([2.0], dtype=mindspore.float32)
        >>> rate_a = Tensor([2.0, 2.0, 2.0], dtype=mindspore.float32)
        >>> concentration_b = Tensor([1.0], dtype=mindspore.float32)
        >>> rate_b = Tensor([1.0, 1.5, 2.0], dtype=mindspore.float32)
        >>>
        >>> # Private interfaces of probability functions corresponding to public interfaces, including
        >>> # `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, and `log_survival`,
        >>> # have the same arguments as follows.
        >>> # Args:
        >>> #     value (Tensor): the value to be evaluated.
        >>> #     concentration (Tensor): the concentration of the distribution. Default: self._concentration.
        >>> #     rate (Tensor): the rate of the distribution. Default: self._rate.
        >>> # Examples of `prob`.
        >>> # Similar calls can be made to other probability functions
        >>> # by replacing 'prob' by the name of the function
        >>> ans = g1.prob(value)
        >>> print(ans.shape)
        (3,)
        >>> # Evaluate with respect to the distribution b.
        >>> ans = g1.prob(value, concentration_b, rate_b)
        >>> print(ans.shape)
        (3,)
        >>> # `concentration` and `rate` must be passed in during function calls for g2.
        >>> ans = g2.prob(value, concentration_a, rate_a)
        >>> print(ans.shape)
        (3,)
        >>> # Functions `mean`, `sd`, `mode`, `var`, and `entropy` have the same arguments.
        >>> # Args:
        >>> #     concentration (Tensor): the concentration of the distribution. Default: self._concentration.
        >>> #     rate (Tensor): the rate of the distribution. Default: self._rate.
        >>> # Example of `mean`, `sd`, `mode`, `var`, and `entropy` are similar.
        >>> ans = g1.mean()
        >>> print(ans.shape)
        (1,)
        >>> ans = g1.mean(concentration_b, rate_b)
        >>> print(ans.shape)
        (3,)
        >>> # `concentration` and `rate` must be passed in during function calls.
        >>> ans = g2.mean(concentration_a, rate_a)
        >>> print(ans.shape)
        (3,)
        >>> # Interfaces of 'kl_loss' and 'cross_entropy' are the same:
        >>> # Args:
        >>> #     dist (str): the type of the distributions. Only "Gamma" is supported.
        >>> #     concentration_b (Tensor): the concentration of distribution b.
        >>> #     rate_b (Tensor): the rate of distribution b.
        >>> #     concentration_a (Tensor): the concentration of distribution a. Default: self._concentration.
        >>> #     rate_a (Tensor): the rate of distribution a. Default: self._rate.
        >>> # Examples of `kl_loss`. `cross_entropy` is similar.
        >>> ans = g1.kl_loss('Gamma', concentration_b, rate_b)
        >>> print(ans.shape)
        (3,)
        >>> ans = g1.kl_loss('Gamma', concentration_b, rate_b, concentration_a, rate_a)
        >>> print(ans.shape)
        (3,)
        >>> # Additional `concentration` and `rate` must be passed in.
        >>> ans = g2.kl_loss('Gamma', concentration_b, rate_b, concentration_a, rate_a)
        >>> print(ans.shape)
        (3,)
        >>> # Examples of `sample`.
        >>> # Args:
        >>> #     shape (tuple): the shape of the sample. Default: ()
        >>> #     concentration (Tensor): the concentration of the distribution. Default: self._concentration.
        >>> #     rate (Tensor): the rate of the distribution. Default: self._rate.
        >>> ans = g1.sample()
        >>> print(ans.shape)
        (1,)
        >>> ans = g1.sample((2,3))
        >>> print(ans.shape)
        (2, 3, 1)
        >>> ans = g1.sample((2,3), concentration_b, rate_b)
        >>> print(ans.shape)
        (2, 3, 3)
        >>> ans = g2.sample((2,3), concentration_a, rate_a)
        >>> print(ans.shape)
        (2, 3, 3)
    """

    def __init__(self,
                 concentration=None,
                 rate=None,
                 seed=None,
                 dtype=mstype.float32,
                 name="Gamma"):
        """
        Constructor of Gamma.
        """
        param = dict(locals())
        param['param_dict'] = {'concentration': concentration, 'rate': rate}
        valid_dtype = mstype.float_type
        Validator.check_type_name(
            "dtype", dtype, valid_dtype, type(self).__name__)

        # As some operators can't accept scalar input, check the type here
        if isinstance(concentration, (int, float)):
            raise TypeError("Input concentration can't be scalar")
        if isinstance(rate, (int, float)):
            raise TypeError("Input rate can't be scalar")

        super(Gamma, self).__init__(seed, dtype, name, param)

        self._concentration = self._add_parameter(
            concentration, 'concentration')
        self._rate = self._add_parameter(rate, 'rate')
        if self._concentration is not None:
            check_greater_zero(self._concentration, "concentration")
        if self._rate is not None:
            check_greater_zero(self._rate, "rate")

        # ops needed for the class
        self.log = log_generic
        self.square = P.Square()
        self.sqrt = P.Sqrt()
        self.squeeze = P.Squeeze(0)
        self.cast = P.Cast()
        self.dtypeop = P.DType()
        self.fill = P.Fill()
        self.shape = P.Shape()
        self.select = P.Select()
        self.greater = P.Greater()
        self.lgamma = nn.LGamma()
        self.digamma = nn.DiGamma()
        self.igamma = nn.IGamma()

    def extend_repr(self):
        """Display instance object as string."""
        if self.is_scalar_batch:
            s = 'concentration = {}, rate = {}'.format(
                self._concentration, self._rate)
        else:
            s = 'batch_shape = {}'.format(self._broadcast_shape)
        return s

    @property
    def concentration(self):
        """
        Return the concentration, also know as the alpha of the Gamma distribution,
        after casting to dtype.

        Output:
            Tensor, the concentration parameter of the distribution.
        """
        return self._concentration

    @property
    def rate(self):
        """
        Return the rate, also know as the beta of the Gamma distribution,
        after casting to dtype.

        Output:
            Tensor, the rate parameter of the distribution.
        """
        return self._rate

    def _get_dist_type(self):
        return "Gamma"

    def _get_dist_args(self, concentration=None, rate=None):
        if concentration is not None:
            self.checktensor(concentration, 'concentration')
        else:
            concentration = self._concentration
        if rate is not None:
            self.checktensor(rate, 'rate')
        else:
            rate = self._rate
        return concentration, rate

    def _mean(self, concentration=None, rate=None):
        """
        The mean of the distribution.
        """
        concentration, rate = self._check_param_type(concentration, rate)
        return concentration / rate

    def _var(self, concentration=None, rate=None):
        """
        The variance of the distribution.
        """
        concentration, rate = self._check_param_type(concentration, rate)
        return concentration / self.square(rate)

    def _sd(self, concentration=None, rate=None):
        """
        The standard deviation of the distribution.
        """
        concentration, rate = self._check_param_type(concentration, rate)
        return self.sqrt(concentration) / rate

    def _mode(self, concentration=None, rate=None):
        """
        The mode of the distribution.
        """
        concentration, rate = self._check_param_type(concentration, rate)
        mode = (concentration - 1.) / rate
        nan = self.fill(self.dtypeop(concentration),
                        self.shape(concentration), np.nan)
        comp = self.greater(concentration, 1.)
        return self.select(comp, mode, nan)

    def _entropy(self, concentration=None, rate=None):
        r"""
        Evaluate entropy.

        .. math::
            H(X) = \alpha - \log(\beta) + \log(\Gamma(\alpha)) + (1 - \alpha) * \digamma(\alpha)
        """
        concentration, rate = self._check_param_type(concentration, rate)
        return concentration - self.log(rate) + self.lgamma(concentration) \
            + (1. - concentration) * self.digamma(concentration)

    def _cross_entropy(self, dist, concentration_b, rate_b, concentration_a=None, rate_a=None):
        r"""
        Evaluate cross entropy between Gamma distributions.

        Args:
            dist (str): Type of the distributions. Should be "Gamma" in this case.
            concentration_b (Tensor): concentration of distribution b.
            rate_b (Tensor): rate of distribution b.
            concentration_a (Tensor): concentration of distribution a. Default: self._concentration.
            rate_a (Tensor): rate of distribution a. Default: self._rate.
        """
        check_distribution_name(dist, 'Gamma')
        return self._entropy(concentration_a, rate_a) +\
            self._kl_loss(dist, concentration_b, rate_b,
                          concentration_a, rate_a)

    def _log_prob(self, value, concentration=None, rate=None):
        r"""
        Evaluate log probability.

        Args:
            value (Tensor): The value to be evaluated.
            concentration (Tensor): The concentration of the distribution. Default: self._concentration.
            rate (Tensor): The rate the distribution. Default: self._rate.

        .. math::
            L(x) = (\alpha - 1) * \log(x) - \beta * x - \log(\gamma(\alpha)) - \alpha * \log(\beta)
        """
        value = self._check_value(value, 'value')
        value = self.cast(value, self.dtype)
        concentration, rate = self._check_param_type(concentration, rate)
        unnormalized_log_prob = (concentration - 1.) * \
            self.log(value) - rate * value
        log_normalization = self.lgamma(
            concentration) - concentration * self.log(rate)
        return unnormalized_log_prob - log_normalization

    def _cdf(self, value, concentration=None, rate=None):
        r"""
        Evaluate the cumulative distribution function on the given value. Note that igamma returns
        the regularized incomplete gamma function, which is what we want for the CDF.

        Args:
            value (Tensor): The value to be evaluated.
            concentration (Tensor): The concentration of the distribution. Default: self._concentration.
            rate (Tensor): The rate the distribution. Default: self._rate.

        .. math::
            cdf(x) = \igamma(\alpha, \beta * x)
        """
        value = self._check_value(value, 'value')
        value = self.cast(value, self.dtype)
        concentration, rate = self._check_param_type(concentration, rate)
        return self.igamma(concentration, rate * value)

    def _kl_loss(self, dist, concentration_b, rate_b, concentration_a=None, rate_a=None):
        r"""
        Evaluate Gamma-Gamma KL divergence, i.e. KL(a||b).

        Args:
            dist (str): The type of the distributions. Should be "Gamma" in this case.
            concentration_b (Tensor): The concentration of distribution b.
            rate_b (Tensor): The rate distribution b.
            concentration_a (Tensor): The concentration of distribution a. Default: self._concentration.
            rate_a (Tensor): The rate distribution a. Default: self._rate.

        .. math::
            KL(a||b) = (\alpha_{a} - \alpha_{b}) * \digamma(\alpha_{a}) + \log(\gamma(\alpha_{b}))
                       - \log(\gamma(\alpha_{a})) + \alpha_{b} * \log(\beta{a}) - \alpha_{b} * \log(\beta{b})
                       + \alpha_{a} * \frac{\beta{b}}{\beta{a} - 1}
        """
        check_distribution_name(dist, 'Gamma')
        concentration_b = self._check_value(concentration_b, 'concentration_b')
        rate_b = self._check_value(rate_b, 'rate_b')
        concentration_b = self.cast(concentration_b, self.parameter_type)
        rate_b = self.cast(rate_b, self.parameter_type)
        concentration_a, rate_a = self._check_param_type(
            concentration_a, rate_a)
        return (concentration_a - concentration_b) * self.digamma(concentration_a) \
            + self.lgamma(concentration_b) - self.lgamma(concentration_a) \
            + concentration_b * self.log(rate_a) - concentration_b * self.log(rate_b) \
            + concentration_a * (rate_b / rate_a - 1.)

    def _sample(self, shape=(), concentration=None, rate=None):
        """
        Sampling.

        Args:
            shape (tuple): The shape of the sample. Default: ().
            concentration (Tensor): The concentration of the samples. Default: self._concentration.
            rate (Tensor): The rate of the samples. Default: self._rate.

        Returns:
            Tensor, with the shape being shape + batch_shape.
        """
        shape = self.checktuple(shape, 'shape')
        concentration, rate = self._check_param_type(concentration, rate)
        batch_shape = self.shape(concentration + rate)
        origin_shape = shape + batch_shape
        if origin_shape == ():
            sample_shape = (1,)
        else:
            sample_shape = origin_shape
        sample_gamma = C.gamma(sample_shape, concentration, rate, self.seed)
        value = self.cast(sample_gamma, self.dtype)
        if origin_shape == ():
            value = self.squeeze(value)
        return value
