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
"""Geometric Distribution"""
import numpy as np
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore._checkparam import Validator
from mindspore.common import dtype as mstype
from .distribution import Distribution
from ._utils.utils import check_prob, check_distribution_name
from ._utils.custom_ops import exp_generic, log_generic


class Geometric(Distribution):
    """
    Geometric Distribution.

    It represents that there are k failures before the first success, namely that there are in total k+1 Bernoulli
    trials when the first success is achieved.

    Args:
        probs (float, list, numpy.ndarray, Tensor): The probability of success.
        seed (int): The seed used in sampling. Global seed is used if it is None. Default: None.
        dtype (mindspore.dtype): The type of the event samples. Default: mstype.int32.
        name (str): The name of the distribution. Default: 'Geometric'.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:
        `probs` must be a proper probability (0 < p < 1).
        `dist_spec_args` is `probs`.

    Examples:
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> import mindspore.nn.probability.distribution as msd
        >>> from mindspore import Tensor
        >>> # To initialize a Geometric distribution of the probability 0.5.
        >>> g1 = msd.Geometric(0.5, dtype=mindspore.int32)
        >>> # A Geometric distribution can be initialized without arguments.
        >>> # In this case, `probs` must be passed in through arguments during function calls.
        >>> g2 = msd.Geometric(dtype=mindspore.int32)
        >>>
        >>> # Here are some tensors used below for testing
        >>> value = Tensor([1, 0, 1], dtype=mindspore.int32)
        >>> probs_a = Tensor([0.6], dtype=mindspore.float32)
        >>> probs_b = Tensor([0.2, 0.5, 0.4], dtype=mindspore.float32)
        >>>
        >>> # Private interfaces of probability functions corresponding to public interfaces, including
        >>> # `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, and `log_survival`, have the same arguments as follows.
        >>> # Args:
        >>> #     value (Tensor): the value to be evaluated.
        >>> #     probs1 (Tensor): the probability of success of a Bernoulli trial. Default: self.probs.
        >>> # Examples of `prob`.
        >>> # Similar calls can be made to other probability functions
        >>> # by replacing `prob` by the name of the function.
        >>> ans = g1.prob(value)
        >>> print(ans.shape)
        (3,)
        >>> # Evaluate with respect to distribution b.
        >>> ans = g1.prob(value, probs_b)
        >>> print(ans.shape)
        (3,)
        >>> # `probs` must be passed in during function calls.
        >>> ans = g2.prob(value, probs_a)
        >>> print(ans.shape)
        (3,)
        >>> # Functions `mean`, `sd`, `var`, and `entropy` have the same arguments.
        >>> # Args:
        >>> #     probs1 (Tensor): the probability of success of a Bernoulli trial. Default: self.probs.
        >>> # Examples of `mean`. `sd`, `var`, and `entropy` are similar.
        >>> ans = g1.mean() # return 1.0
        >>> print(ans.shape)
        ()
        >>> ans = g1.mean(probs_b)
        >>> print(ans.shape)
        (3,)
        >>> # Probs must be passed in during function calls
        >>> ans = g2.mean(probs_a)
        >>> print(ans.shape)
        (1,)
        >>> # Interfaces of 'kl_loss' and 'cross_entropy' are the same.
        >>> # Args:
        >>> #     dist (str): the name of the distribution. Only 'Geometric' is supported.
        >>> #     probs1_b (Tensor): the probability of success of a Bernoulli trial of distribution b.
        >>> #     probs1_a (Tensor): the probability of success of a Bernoulli trial of distribution a. Default: self.probs.
        >>> # Examples of `kl_loss`. `cross_entropy` is similar.
        >>> ans = g1.kl_loss('Geometric', probs_b)
        >>> print(ans.shape)
        (3,)
        >>> ans = g1.kl_loss('Geometric', probs_b, probs_a)
        >>> print(ans.shape)
        (3,)
        >>> # An additional `probs` must be passed in.
        >>> ans = g2.kl_loss('Geometric', probs_b, probs_a)
        >>> print(ans.shape)
        (3,)
        >>> # Examples of `sample`.
        >>> # Args:
        >>> #     shape (tuple): the shape of the sample. Default: ()
        >>> #     probs1 (Tensor): the probability of success of a Bernoulli trial. Default: self.probs.
        >>> ans = g1.sample()
        >>> print(ans.shape)
        ()
        >>> ans = g1.sample((2,3))
        >>> print(ans.shape)
        (2, 3)
        >>> ans = g1.sample((2,3), probs_b)
        >>> print(ans.shape)
        (2, 3, 3)
        >>> ans = g2.sample((2,3), probs_a)
        >>> print(ans.shape)
        (2, 3, 1)
    """

    def __init__(self,
                 probs=None,
                 seed=None,
                 dtype=mstype.int32,
                 name="Geometric"):
        """
        Constructor of Geometric distribution.
        """
        param = dict(locals())
        param['param_dict'] = {'probs': probs}
        valid_dtype = mstype.int_type + mstype.uint_type + mstype.float_type
        Validator.check_type_name("dtype", dtype, valid_dtype, type(self).__name__)
        super(Geometric, self).__init__(seed, dtype, name, param)

        self._probs = self._add_parameter(probs, 'probs')
        if self._probs is not None:
            check_prob(self.probs)

        self.minval = np.finfo(np.float).tiny

        # ops needed for the class
        self.exp = exp_generic
        self.log = log_generic
        self.squeeze = P.Squeeze(0)
        self.cast = P.Cast()
        self.const = P.ScalarToArray()
        self.dtypeop = P.DType()
        self.fill = P.Fill()
        self.floor = P.Floor()
        self.issubclass = P.IsSubClass()
        self.less = P.Less()
        self.pow = P.Pow()
        self.select = P.Select()
        self.shape = P.Shape()
        self.sq = P.Square()
        self.uniform = C.uniform

    def extend_repr(self):
        if self.is_scalar_batch:
            s = f'probs = {self.probs}'
        else:
            s = f'batch_shape = {self._broadcast_shape}'
        return s

    @property
    def probs(self):
        """
        Return the probability of success of the Bernoulli trial,
        after casting to dtype.
        """
        return self._probs

    def _get_dist_type(self):
        return "Geometric"

    def _get_dist_args(self, probs1=None):
        if probs1 is not None:
            self.checktensor(probs1, 'probs')
        else:
            probs1 = self.probs
        return (probs1,)

    def _mean(self, probs1=None):
        r"""
        .. math::
            MEAN(Geo) = \fratc{1 - probs1}{probs1}
        """
        probs1 = self._check_param_type(probs1)
        return (1. - probs1) / probs1

    def _mode(self, probs1=None):
        r"""
        .. math::
            MODE(Geo) = 0
        """
        probs1 = self._check_param_type(probs1)
        return self.fill(self.dtype, self.shape(probs1), 0.)

    def _var(self, probs1=None):
        r"""
        .. math::
            VAR(Geo) = \frac{1 - probs1}{probs1 ^ {2}}
        """
        probs1 = self._check_param_type(probs1)
        return (1.0 - probs1) / self.sq(probs1)

    def _entropy(self, probs1=None):
        r"""
        .. math::
            H(Geo) = \frac{-1 * probs0 \log_2 (1-probs0)\ - prob1 * \log_2 (1-probs1)\ }{probs1}
        """
        probs1 = self._check_param_type(probs1)
        probs0 = 1.0 - probs1
        return (-probs0 * self.log(probs0) - probs1 * self.log(probs1)) / probs1

    def _cross_entropy(self, dist, probs1_b, probs1=None):
        r"""
        Evaluate cross entropy between Geometric distributions.

        Args:
            dist (str): The type of the distributions. Should be "Geometric" in this case.
            probs1_b (Tensor): The probability of success of distribution b.
            probs1_a (Tensor): The probability of success of distribution a. Default: self.probs.
        """
        check_distribution_name(dist, 'Geometric')
        return self._entropy(probs1) + self._kl_loss(dist, probs1_b, probs1)

    def _prob(self, value, probs1=None):
        r"""
        Probability mass function of Geometric distributions.

        Args:
            value (Tensor): A Tensor composed of only natural numbers.
            probs (Tensor): The probability of success. Default: self.probs.

        .. math::
            pmf(k) = probs0 ^k * probs1 if k >= 0;
            pmf(k) = 0 if k < 0.
        """
        value = self._check_value(value, 'value')
        value = self.cast(value, self.parameter_type)
        value = self.floor(value)
        probs1 = self._check_param_type(probs1)
        pmf = self.exp(self.log(1.0 - probs1) * value + self.log(probs1))
        zeros = self.fill(self.dtypeop(pmf), self.shape(pmf), 0.0)
        comp = self.less(value, zeros)
        return self.select(comp, zeros, pmf)

    def _cdf(self, value, probs1=None):
        r"""
        Cumulative distribution function (cdf) of Geometric distributions.

        Args:
            value (Tensor): A Tensor composed of only natural numbers.
            probs (Tensor): The probability of success. Default: self.probs.

        .. math::
            cdf(k) = 1 - probs0 ^ (k+1) if k >= 0;
            cdf(k) = 0 if k < 0.

        """
        value = self._check_value(value, 'value')
        value = self.cast(value, self.parameter_type)
        value = self.floor(value)
        probs1 = self._check_param_type(probs1)
        probs0 = 1.0 - probs1
        cdf = 1.0 - self.pow(probs0, value + 1.0)
        zeros = self.fill(self.dtypeop(cdf), self.shape(cdf), 0.0)
        comp = self.less(value, zeros)
        return self.select(comp, zeros, cdf)

    def _kl_loss(self, dist, probs1_b, probs1=None):
        r"""
        Evaluate Geometric-Geometric kl divergence, i.e. KL(a||b).

        Args:
            dist (str): The type of the distributions. Should be "Geometric" in this case.
            probs1_b (Tensor): The probability of success of distribution b.
            probs1_a (Tensor): The probability of success of distribution a. Default: self.probs.

        .. math::
            KL(a||b) = \log(\frac{probs1_a}{probs1_b}) + \frac{probs0_a}{probs1_a} * \log(\frac{probs0_a}{probs0_b})
        """
        check_distribution_name(dist, 'Geometric')
        probs1_b = self._check_value(probs1_b, 'probs1_b')
        probs1_b = self.cast(probs1_b, self.parameter_type)
        probs1_a = self._check_param_type(probs1)
        probs0_a = 1.0 - probs1_a
        probs0_b = 1.0 - probs1_b
        return self.log(probs1_a / probs1_b) + (probs0_a / probs1_a) * self.log(probs0_a / probs0_b)

    def _sample(self, shape=(), probs1=None):
        """
        Sampling.

        Args:
            shape (tuple): The shape of the sample. Default: ().
            probs (Tensor): The probability of success. Default: self.probs.

        Returns:
            Tensor,  with the shape being shape + batch_shape.
        """
        shape = self.checktuple(shape, 'shape')
        probs1 = self._check_param_type(probs1)
        origin_shape = shape + self.shape(probs1)
        if origin_shape == ():
            sample_shape = (1,)
        else:
            sample_shape = origin_shape
        minval = self.const(self.minval)
        maxval = self.const(1.0)
        sample_uniform = self.uniform(sample_shape, minval, maxval, self.seed)
        sample = self.floor(self.log(sample_uniform) / self.log(1.0 - probs1))
        value = self.cast(sample, self.dtype)
        if origin_shape == ():
            value = self.squeeze(value)
        return value
