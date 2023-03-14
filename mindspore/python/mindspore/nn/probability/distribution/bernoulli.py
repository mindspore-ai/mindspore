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
"""Bernoulli Distribution"""
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore._checkparam import Validator
from .distribution import Distribution
from ._utils.utils import check_prob, check_distribution_name, clamp_probs
from ._utils.custom_ops import exp_generic, log_generic


class Bernoulli(Distribution):
    r"""
    Bernoulli Distribution.
    A Bernoulli Distribution is a discrete distribution with the range :math:`\{0, 1\}`
    and the probability mass function as :math:`P(X = 0) = p, P(X = 1) = 1-p`.

    Args:
        probs (float, list, numpy.ndarray, Tensor): The probability of that the outcome is 1. Default: None.
        seed (int): The seed used in sampling. The global seed is used if it is None. Default: None.
        dtype (mindspore.dtype): The type of the event samples. Default: mstype.int32.
        name (str): The name of the distribution. Default: 'Bernoulli'.

    Note:
        `probs` must be a proper probability (0 < p < 1).
        `dist_spec_args` is `probs`.

    Raises:
        ValueError: When p <= 0 or p >=1.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> import mindspore.nn.probability.distribution as msd
        >>> from mindspore import Tensor
        >>> # To initialize a Bernoulli distribution of the probability 0.5.
        >>> b1 = msd.Bernoulli(0.5, dtype=mindspore.int32)
        >>> # A Bernoulli distribution can be initialized without arguments.
        >>> # In this case, `probs` must be passed in through arguments during function calls.
        >>> b2 = msd.Bernoulli(dtype=mindspore.int32)
        >>> # Here are some tensors used below for testing
        >>> value = Tensor([1, 0, 1], dtype=mindspore.int32)
        >>> probs_a = Tensor([0.6], dtype=mindspore.float32)
        >>> probs_b = Tensor([0.2, 0.3, 0.4], dtype=mindspore.float32)
        >>> # Private interfaces of probability functions corresponding to public interfaces, including
        >>> # `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, and `log_survival`, are the same as follows.
        >>> # Args:
        >>> #     value (Tensor): the value to be evaluated.
        >>> #     probs1 (Tensor): the probability of success. Default: self.probs.
        >>> # Examples of `prob`.
        >>> # Similar calls can be made to other probability functions
        >>> # by replacing `prob` by the name of the function.
        >>> ans = b1.prob(value)
        >>> print(ans.shape)
        (3,)
        >>> # Evaluate `prob` with respect to distribution b.
        >>> ans = b1.prob(value, probs_b)
        >>> print(ans.shape)
        (3,)
        >>> # `probs` must be passed in during function calls.
        >>> ans = b2.prob(value, probs_a)
        >>> print(ans.shape)
        (3,)
        >>> # Functions `mean`, `sd`, `var`, and `entropy` have the same arguments.
        >>> # Args:
        >>> #     probs1 (Tensor): the probability of success. Default: self.probs.
        >>> # Examples of `mean`. `sd`, `var`, and `entropy` are similar.
        >>> ans = b1.mean() # return 0.5
        >>> print(ans.shape)
        ()
        >>> ans = b1.mean(probs_b) # return probs_b
        >>> print(ans.shape)
        (3,)
        >>> # `probs` must be passed in during function calls.
        >>> ans = b2.mean(probs_a)
        >>> print(ans.shape)
        (1,)
        >>> # Interfaces of `kl_loss` and `cross_entropy` are the same as follows:
        >>> # Args:
        >>> #     dist (str): the name of the distribution. Only 'Bernoulli' is supported.
        >>> #     probs1_b (Tensor): the probability of success of distribution b.
        >>> #     probs1_a (Tensor): the probability of success of distribution a. Default: self.probs.
        >>> # Examples of `kl_loss`. `cross_entropy` is similar.
        >>> ans = b1.kl_loss('Bernoulli', probs_b)
        >>> print(ans.shape)
        (3,)
        >>> ans = b1.kl_loss('Bernoulli', probs_b, probs_a)
        >>> print(ans.shape)
        (3,)
        >>> # An additional `probs_a` must be passed in.
        >>> ans = b2.kl_loss('Bernoulli', probs_b, probs_a)
        >>> print(ans.shape)
        (3,)
        >>> # Examples of `sample`.
        >>> # Args:
        >>> #     shape (tuple): the shape of the sample. Default: ().
        >>> #     probs1 (Tensor): the probability of success. Default: self.probs.
        >>> ans = b1.sample()
        >>> print(ans.shape)
        ()
        >>> ans = b1.sample((2,3))
        >>> print(ans.shape)
        (2, 3)
        >>> ans = b1.sample((2,3), probs_b)
        >>> print(ans.shape)
        (2, 3, 3)
        >>> ans = b2.sample((2,3), probs_a)
        >>> print(ans.shape)
        (2, 3, 1)
    """

    def __init__(self,
                 probs=None,
                 seed=None,
                 dtype=mstype.int32,
                 name="Bernoulli"):
        """
        Constructor of Bernoulli.
        """
        param = dict(locals())
        param['param_dict'] = {'probs': probs}
        valid_dtype = mstype.int_type + mstype.uint_type + mstype.float_type
        Validator.check_type_name(
            "dtype", dtype, valid_dtype, type(self).__name__)
        super(Bernoulli, self).__init__(seed, dtype, name, param)

        self._probs = self._add_parameter(probs, 'probs')
        if self._probs is not None:
            check_prob(self.probs)

        # ops needed for the class
        self.exp = exp_generic
        self.log = log_generic
        self.squeeze = P.Squeeze(0)
        self.cast = P.Cast()
        self.const = P.ScalarToTensor()
        self.floor = P.Floor()
        self.fill = P.Fill()
        self.less = P.Less()
        self.shape = P.Shape()
        self.select = P.Select()
        self.uniform = C.uniform

    def extend_repr(self):
        """Display instance object as string."""
        if self.is_scalar_batch:
            s = 'probs = {}'.format(self.probs)
        else:
            s = 'batch_shape = {}'.format(self._broadcast_shape)
        return s

    @property
    def probs(self):
        """
        Return the probability of that the outcome is 1
        after casting to dtype.

        Output:
            Tensor, the probs of the distribution.
        """
        return self._probs

    def _get_dist_type(self):
        return "Bernoulli"

    def _get_dist_args(self, probs1=None):
        if probs1 is not None:
            self.checktensor(probs1, 'probs')
        else:
            probs1 = self.probs
        return (probs1,)

    def _mean(self, probs1=None):
        r"""
        .. math::
            MEAN(B) = probs1
        """
        probs1 = self._check_param_type(probs1)
        return probs1

    def _mode(self, probs1=None):
        r"""
        .. math::
            MODE(B) = 1 if probs1 > 0.5 else = 0
        """
        probs1 = self._check_param_type(probs1)
        zeros = self.fill(self.dtype, self.shape(probs1), 0.0)
        ones = self.fill(self.dtype, self.shape(probs1), 1.0)
        comp = self.less(0.5, probs1)
        return self.select(comp, ones, zeros)

    def _var(self, probs1=None):
        r"""
        .. math::
            VAR(B) = probs1 * probs0
        """
        probs1 = self._check_param_type(probs1)
        probs0 = 1.0 - probs1
        return self.exp(self.log(probs0) + self.log(probs1))

    def _entropy(self, probs1=None):
        r"""
        .. math::
            H(B) = -probs0 * \log(probs0) - probs1 * \log(probs1)
        """
        probs1 = self._check_param_type(probs1)
        probs0 = 1.0 - probs1
        return -(probs0 * self.log(probs0)) - (probs1 * self.log(probs1))

    def _cross_entropy(self, dist, probs1_b, probs1=None):
        """
        Evaluate cross entropy between Bernoulli distributions.

        Args:
            dist (str): The type of the distributions. Should be "Bernoulli" in this case.
            probs1_b (Tensor): `probs1` of distribution b.
            probs1_a (Tensor): `probs1` of distribution a. Default: self.probs.
        """
        check_distribution_name(dist, 'Bernoulli')
        return self._entropy(probs1) + self._kl_loss(dist, probs1_b, probs1)

    def _log_prob(self, value, probs1=None):
        r"""
        Log probability mass function of Bernoulli distributions.

        Args:
            value (Tensor): A Tensor composed of only zeros and ones.
            probs (Tensor): The probability of outcome is 1. Default: self.probs.

        .. math::
            pmf(k) = probs1 if k = 1;
            pmf(k) = probs0 if k = 0;
        """
        value = self._check_value(value, 'value')
        value = self.cast(value, self.parameter_type)
        probs1 = self._check_param_type(probs1)

        # clamp value for numerical stability
        probs1 = clamp_probs(probs1)
        probs0 = 1.0 - probs1
        return self.log(probs1) * value + self.log(probs0) * (1.0 - value)

    def _cdf(self, value, probs1=None):
        r"""
        Cumulative distribution function (cdf) of Bernoulli distributions.

        Args:
            value (Tensor): The value to be evaluated.
            probs (Tensor): The probability of that the outcome is 1. Default: self.probs.

        .. math::
            cdf(k) = 0 if k < 0;
            cdf(k) = probs0 if 0 <= k <1;
            cdf(k) = 1 if k >=1;
        """
        value = self._check_value(value, 'value')
        value = self.cast(value, self.parameter_type)
        value = self.floor(value)
        probs1 = self._check_param_type(probs1)
        broadcast_shape_tensor = value * probs1
        value = self.broadcast(value, broadcast_shape_tensor)
        probs0 = self.broadcast((1.0 - probs1), broadcast_shape_tensor)
        comp_zero = self.less(value, 0.0)
        comp_one = self.less(value, 1.0)
        zeros = self.fill(self.parameter_type, self.shape(
            broadcast_shape_tensor), 0.0)
        ones = self.fill(self.parameter_type, self.shape(
            broadcast_shape_tensor), 1.0)
        less_than_zero = self.select(comp_zero, zeros, probs0)
        return self.select(comp_one, less_than_zero, ones)

    def _kl_loss(self, dist, probs1_b, probs1=None):
        r"""
        Evaluate bernoulli-bernoulli kl divergence, i.e. KL(a||b).

        Args:
            dist (str): The type of the distributions. Should be "Bernoulli" in this case.
            probs1_b (Union[Tensor, numbers.Number]): `probs1` of distribution b.
            probs1_a (Union[Tensor, numbers.Number]): `probs1` of distribution a. Default: self.probs.

        .. math::
            KL(a||b) = probs1_a * \log(\frac{probs1_a}{probs1_b}) +
                       probs0_a * \log(\frac{probs0_a}{probs0_b})
        """
        check_distribution_name(dist, 'Bernoulli')
        probs1_b = self._check_value(probs1_b, 'probs1_b')
        probs1_b = self.cast(probs1_b, self.parameter_type)
        probs1_a = self._check_param_type(probs1)
        probs0_a = 1.0 - probs1_a
        probs0_b = 1.0 - probs1_b
        return probs1_a * self.log(probs1_a / probs1_b) + probs0_a * self.log(probs0_a / probs0_b)

    def _sample(self, shape=(), probs1=None):
        """
        Sampling.

        Args:
            shape (tuple): The shape of the sample. Default: ().
            probs1 (Tensor): `probs1` of the samples. Default: self.probs.

        Returns:
            Tensor, shape is shape + batch_shape.
        """
        shape = self.checktuple(shape, 'shape')
        probs1 = self._check_param_type(probs1)
        origin_shape = shape + self.shape(probs1)
        if origin_shape == ():
            sample_shape = (1,)
        else:
            sample_shape = origin_shape
        l_zero = self.const(0.0, mstype.float32)
        h_one = self.const(1.0, mstype.float32)
        sample_uniform = self.uniform(sample_shape, l_zero, h_one, self.seed)
        sample = self.less(sample_uniform, probs1)
        value = self.cast(sample, self.dtype)
        if origin_shape == ():
            value = self.squeeze(value)
        return value
