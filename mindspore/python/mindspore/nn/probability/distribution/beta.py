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
"""Beta Distribution"""
import numpy as np
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
import mindspore.nn as nn
from mindspore import _checkparam as Validator
from mindspore.common import dtype as mstype
from .distribution import Distribution
from ._utils.utils import check_greater_zero, check_distribution_name
from ._utils.custom_ops import log_generic


class Beta(Distribution):
    r"""
    Beta distribution.
    A Beta distributio is a continuous distribution with the range :math:`[0, 1]` and the probability density function:

    .. math::
        f(x, \alpha, \beta) = x^\alpha (1-x)^{\beta - 1} / B(\alpha, \beta),

    where :math:`B` is the Beta function.

    Args:
        concentration1 (int, float, list, numpy.ndarray, Tensor): The concentration1,
          also know as alpha of the Beta distribution. Default: ``None`` .
        concentration0 (int, float, list, numpy.ndarray, Tensor): The concentration0, also know as
          beta of the Beta distribution. Default: ``None`` .
        seed (int): The seed used in sampling. The global seed is used if it is None. Default: ``None`` .
        dtype (mindspore.dtype): The type of the event samples. Default: ``mstype.float32`` .
        name (str): The name of the distribution. Default: ``'Beta'`` .

    Note:
        - `concentration1` and `concentration0` must be greater than zero.
        - `dist_spec_args` are `concentration1` and `concentration0`.
        - `dtype` must be a float type because Beta distributions are continuous.

    Raises:
        ValueError: When concentration1 <= 0 or concentration0 >=1.
        TypeError: When the input `dtype` is not a subclass of float.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> import mindspore.nn.probability.distribution as msd
        >>> from mindspore import Tensor
        >>> # To initialize a Beta distribution of the concentration1 3.0 and the concentration0 4.0.
        >>> b1 = msd.Beta([3.0], [4.0], dtype=mindspore.float32)
        >>> # A Beta distribution can be initialized without arguments.
        >>> # In this case, `concentration1` and `concentration0` must be passed in through arguments.
        >>> b2 = msd.Beta(dtype=mindspore.float32)
        >>> # Here are some tensors used below for testing
        >>> value = Tensor([0.1, 0.5, 0.8], dtype=mindspore.float32)
        >>> concentration1_a = Tensor([2.0], dtype=mindspore.float32)
        >>> concentration0_a = Tensor([2.0, 2.0, 2.0], dtype=mindspore.float32)
        >>> concentration1_b = Tensor([1.0], dtype=mindspore.float32)
        >>> concentration0_b = Tensor([1.0, 1.5, 2.0], dtype=mindspore.float32)
        >>> # Private interfaces of probability functions corresponding to public interfaces, including
        >>> # `prob` and `log_prob`, have the same arguments as follows.
        >>> # Args:
        >>> #     value (Tensor): the value to be evaluated.
        >>> #     concentration1 (Tensor): the concentration1 of the distribution. Default: self._concentration1.
        >>> #     concentration0 (Tensor): the concentration0 of the distribution. Default: self._concentration0.
        >>> # Examples of `prob`.
        >>> # Similar calls can be made to other probability functions
        >>> # by replacing 'prob' by the name of the function
        >>> ans = b1.prob(value)
        >>> print(ans.shape)
        (3,)
        >>> # Evaluate with respect to the distribution b.
        >>> ans = b1.prob(value, concentration1_b, concentration0_b)
        >>> print(ans.shape)
        (3,)
        >>> # `concentration1` and `concentration0` must be passed in during function calls
        >>> ans = b2.prob(value, concentration1_a, concentration0_a)
        >>> print(ans.shape)
        (3,)
        >>> # Functions `mean`, `sd`, `mode`, `var`, and `entropy` have the same arguments.
        >>> # Args:
        >>> #     concentration1 (Tensor): the concentration1 of the distribution. Default: self._concentration1.
        >>> #     concentration0 (Tensor): the concentration0 of the distribution. Default: self._concentration0.
        >>> # Example of `mean`, `sd`, `mode`, `var`, and `entropy` are similar.
        >>> ans = b1.mean()
        >>> print(ans.shape)
        (1,)
        >>> ans = b1.mean(concentration1_b, concentration0_b)
        >>> print(ans.shape)
        (3,)
        >>> # `concentration1` and `concentration0` must be passed in during function calls.
        >>> ans = b2.mean(concentration1_a, concentration0_a)
        >>> print(ans.shape)
        (3,)
        >>> # Interfaces of 'kl_loss' and 'cross_entropy' are the same:
        >>> # Args:
        >>> #     dist (str): the type of the distributions. Only "Beta" is supported.
        >>> #     concentration1_b (Tensor): the concentration1 of distribution b.
        >>> #     concentration0_b (Tensor): the concentration0 of distribution b.
        >>> #     concentration1_a (Tensor): the concentration1 of distribution a.
        >>> #       Default: self._concentration1.
        >>> #     concentration0_a (Tensor): the concentration0 of distribution a.
        >>> #       Default: self._concentration0.
        >>> # Examples of `kl_loss`. `cross_entropy` is similar.
        >>> ans = b1.kl_loss('Beta', concentration1_b, concentration0_b)
        >>> print(ans.shape)
        (3,)
        >>> ans = b1.kl_loss('Beta', concentration1_b, concentration0_b, concentration1_a, concentration0_a)
        >>> print(ans.shape)
        (3,)
        >>> # Additional `concentration1` and `concentration0` must be passed in.
        >>> ans = b2.kl_loss('Beta', concentration1_b, concentration0_b, concentration1_a, concentration0_a)
        >>> print(ans.shape)
        (3,)
        >>> # Examples of `sample`.
        >>> # Args:
        >>> #     shape (tuple): the shape of the sample. Default: ()
        >>> #     concentration1 (Tensor): the concentration1 of the distribution. Default: self._concentration1.
        >>> #     concentration0 (Tensor): the concentration0 of the distribution. Default: self._concentration0.
        >>> ans = b1.sample()
        >>> print(ans.shape)
        (1,)
        >>> ans = b1.sample((2,3))
        >>> print(ans.shape)
        (2, 3, 1)
        >>> ans = b1.sample((2,3), concentration1_b, concentration0_b)
        >>> print(ans.shape)
        (2, 3, 3)
        >>> ans = b2.sample((2,3), concentration1_a, concentration0_a)
        >>> print(ans.shape)
        (2, 3, 3)
    """

    def __init__(self,
                 concentration1=None,
                 concentration0=None,
                 seed=None,
                 dtype=mstype.float32,
                 name="Beta"):
        """
        Constructor of Beta.
        """
        param = dict(locals())
        param['param_dict'] = {
            'concentration1': concentration1, 'concentration0': concentration0}

        valid_dtype = mstype.float_type
        Validator.check_type_name(
            "dtype", dtype, valid_dtype, type(self).__name__)

        # As some operators can't accept scalar input, check the type here
        if isinstance(concentration0, float):
            raise TypeError("Input concentration0 can't be scalar")
        if isinstance(concentration1, float):
            raise TypeError("Input concentration1 can't be scalar")

        super(Beta, self).__init__(seed, dtype, name, param)

        self._concentration1 = self._add_parameter(
            concentration1, 'concentration1')
        self._concentration0 = self._add_parameter(
            concentration0, 'concentration0')
        if self._concentration1 is not None:
            check_greater_zero(self._concentration1, "concentration1")
        if self._concentration0 is not None:
            check_greater_zero(self._concentration0, "concentration0")

        # ops needed for the class
        self.log = log_generic
        self.log1p = P.Log1p()
        self.neg = P.Neg()
        self.pow = P.Pow()
        self.squeeze = P.Squeeze(0)
        self.cast = P.Cast()
        self.shape = P.Shape()
        self.select = P.Select()
        self.logicaland = P.LogicalAnd()
        self.greater = P.Greater()
        self.digamma = nn.DiGamma()
        self.lbeta = nn.LBeta()

    def extend_repr(self):
        """Display instance object as string."""
        if self.is_scalar_batch:
            s = 'concentration1 = {}, concentration0 = {}'.format(
                self._concentration1, self._concentration0)
        else:
            s = 'batch_shape = {}'.format(self._broadcast_shape)
        return s

    @property
    def concentration1(self):
        """
        Return the concentration1, also know as the alpha of the Beta distribution,
        after casting to dtype.

        Output:
            Tensor, the concentration1 parameter of the distribution.
        """
        return self._concentration1

    @property
    def concentration0(self):
        """
        Return the concentration0, also know as the beta of the Beta distribution,
        after casting to dtype.

        Output:
            Tensor, the concentration2 parameter of the distribution.
        """
        return self._concentration0

    def _get_dist_type(self):
        return "Beta"

    def _get_dist_args(self, concentration1=None, concentration0=None):
        if concentration1 is not None:
            self.checktensor(concentration1, 'concentration1')
        else:
            concentration1 = self._concentration1
        if concentration0 is not None:
            self.checktensor(concentration0, 'concentration0')
        else:
            concentration0 = self._concentration0
        return concentration1, concentration0

    def _mean(self, concentration1=None, concentration0=None):
        """
        The mean of the distribution.
        """
        concentration1, concentration0 = self._check_param_type(
            concentration1, concentration0)
        return concentration1 / (concentration1 + concentration0)

    def _var(self, concentration1=None, concentration0=None):
        """
        The variance of the distribution.
        """
        concentration1, concentration0 = self._check_param_type(
            concentration1, concentration0)
        total_concentration = concentration1 + concentration0
        return concentration1 * concentration0 / (self.pow(total_concentration, 2) * (total_concentration + 1.))

    def _mode(self, concentration1=None, concentration0=None):
        """
        The mode of the distribution.
        """
        concentration1, concentration0 = self._check_param_type(
            concentration1, concentration0)
        comp1 = self.greater(concentration1, 1.)
        comp2 = self.greater(concentration0, 1.)
        cond = self.logicaland(comp1, comp2)
        batch_shape = self.shape(concentration1 + concentration0)
        nan = F.fill(self.dtype, batch_shape, np.nan)
        mode = (concentration1 - 1.) / (concentration1 + concentration0 - 2.)
        return self.select(cond, mode, nan)

    def _entropy(self, concentration1=None, concentration0=None):
        r"""
        Evaluate entropy.

        .. math::
            H(X) = \log(\Beta(\alpha, \beta)) - (\alpha - 1) * \digamma(\alpha)
                   - (\beta - 1) * \digamma(\beta) + (\alpha + \beta - 2) * \digamma(\alpha + \beta)
        """
        concentration1, concentration0 = self._check_param_type(
            concentration1, concentration0)
        total_concentration = concentration1 + concentration0
        return self.lbeta(concentration1, concentration0) \
            - (concentration1 - 1.) * self.digamma(concentration1) \
            - (concentration0 - 1.) * self.digamma(concentration0) \
            + (total_concentration - 2.) * self.digamma(total_concentration)

    def _cross_entropy(self, dist, concentration1_b, concentration0_b, concentration1_a=None, concentration0_a=None):
        r"""
        Evaluate cross entropy between Beta distributions.

        Args:
            dist (str): Type of the distributions. Should be "Beta" in this case.
            concentration1_b (Tensor): concentration1 of distribution b.
            concentration0_b (Tensor): concentration0 of distribution b.
            concentration1_a (Tensor): concentration1 of distribution a. Default: self._concentration1.
            concentration0_a (Tensor): concentration0 of distribution a. Default: self._concentration0.
        """
        check_distribution_name(dist, 'Beta')
        return self._entropy(concentration1_a, concentration0_a) \
            + self._kl_loss(dist, concentration1_b, concentration0_b,
                            concentration1_a, concentration0_a)

    def _log_prob(self, value, concentration1=None, concentration0=None):
        r"""
        Evaluate log probability.

        Args:
            value (Tensor): The value to be evaluated.
            concentration1 (Tensor): The concentration1 of the distribution. Default: self._concentration1.
            concentration0 (Tensor): The concentration0 the distribution. Default: self._concentration0.

        .. math::
            L(x) = (\alpha - 1) * \log(x) + (\beta - 1) * \log(1 - x) - \log(\Beta(\alpha, \beta))
        """
        value = self._check_value(value, 'value')
        value = self.cast(value, self.dtype)
        concentration1, concentration0 = self._check_param_type(
            concentration1, concentration0)
        log_unnormalized_prob = (concentration1 - 1.) * self.log(value) \
            + (concentration0 - 1.) * self.log1p(self.neg(value))
        return log_unnormalized_prob - self.lbeta(concentration1, concentration0)

    def _kl_loss(self, dist, concentration1_b, concentration0_b, concentration1_a=None, concentration0_a=None):
        r"""
        Evaluate Beta-Beta KL divergence, i.e. KL(a||b).

        Args:
            dist (str): The type of the distributions. Should be "Beta" in this case.
            concentration1_b (Tensor): The concentration1 of distribution b.
            concentration0_b (Tensor): The concentration0 distribution b.
            concentration1_a (Tensor): The concentration1 of distribution a. Default: self._concentration1.
            concentration0_a (Tensor): The concentration0 distribution a. Default: self._concentration0.

        .. math::
            KL(a||b) = \log(\Beta(\alpha_{b}, \beta_{b})) - \log(\Beta(\alpha_{a}, \beta_{a}))
                       - \digamma(\alpha_{a}) * (\alpha_{b} - \alpha_{a})
                       - \digamma(\beta_{a}) * (\beta_{b} - \beta_{a})
                       + \digamma(\alpha_{a} + \beta_{a}) * (\alpha_{b} + \beta_{b} - \alpha_{a} - \beta_{a})
        """
        check_distribution_name(dist, 'Beta')
        concentration1_b = self._check_value(
            concentration1_b, 'concentration1_b')
        concentration0_b = self._check_value(
            concentration0_b, 'concentration0_b')
        concentration1_b = self.cast(concentration1_b, self.parameter_type)
        concentration0_b = self.cast(concentration0_b, self.parameter_type)
        concentration1_a, concentration0_a = self._check_param_type(
            concentration1_a, concentration0_a)
        total_concentration_a = concentration1_a + concentration0_a
        total_concentration_b = concentration1_b + concentration0_b
        log_normalization_a = self.lbeta(concentration1_a, concentration0_a)
        log_normalization_b = self.lbeta(concentration1_b, concentration0_b)
        return (log_normalization_b - log_normalization_a) \
            - (self.digamma(concentration1_a) * (concentration1_b - concentration1_a)) \
            - (self.digamma(concentration0_a) * (concentration0_b - concentration0_a)) \
            + (self.digamma(total_concentration_a) *
               (total_concentration_b - total_concentration_a))

    def _sample(self, shape=(), concentration1=None, concentration0=None):
        """
        Sampling.

        Args:
            shape (tuple): The shape of the sample. Default: ().
            concentration1 (Tensor): The concentration1 of the samples. Default: self._concentration1.
            concentration0 (Tensor): The concentration0 of the samples. Default: self._concentration0.

        Returns:
            Tensor, with the shape being shape + batch_shape.
        """
        shape = self.checktuple(shape, 'shape')
        concentration1, concentration0 = self._check_param_type(
            concentration1, concentration0)
        batch_shape = self.shape(concentration1 + concentration0)
        origin_shape = shape + batch_shape
        if origin_shape == ():
            sample_shape = (1,)
        else:
            sample_shape = origin_shape
        ones = F.fill(self.dtype, sample_shape, 1.0)
        sample_gamma1 = C.gamma(
            sample_shape, alpha=concentration1, beta=ones, seed=self.seed)
        sample_gamma2 = C.gamma(
            sample_shape, alpha=concentration0, beta=ones, seed=self.seed)
        sample_beta = sample_gamma1 / (sample_gamma1 + sample_gamma2)
        value = self.cast(sample_beta, self.dtype)
        if origin_shape == ():
            value = self.squeeze(value)
        return value
