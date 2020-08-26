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
from mindspore.common import dtype as mstype
from .distribution import Distribution
from ._utils.utils import cast_to_tensor, check_prob, check_type, check_distribution_name,\
    raise_none_error
from ._utils.custom_ops import exp_generic, log_generic


class Geometric(Distribution):
    """
    Geometric Distribution.
    It represents k+1 Bernoulli trials needed to get one success, k is the number of failures.

    Args:
        probs (float, list, numpy.ndarray, Tensor, Parameter): probability of success.
        seed (int): seed to use in sampling. Default: 0.
        dtype (mindspore.dtype): type of the distribution. Default: mstype.int32.
        name (str): name of the distribution. Default: Geometric.

    Note:
        probs should be proper probabilities (0 < p < 1).
        Dist_spec_args is probs.

    Examples:
        >>> # To initialize a Geometric distribution of prob 0.5
        >>> import mindspore.nn.probability.distribution as msd
        >>> n = msd.Geometric(0.5, dtype=mstype.int32)
        >>>
        >>> # The following creates two independent Geometric distributions
        >>> n = msd.Geometric([0.5, 0.5], dtype=mstype.int32)
        >>>
        >>> # A Geometric distribution can be initilized without arguments
        >>> # In this case, probs must be passed in through args during function calls.
        >>> n = msd.Geometric(dtype=mstype.int32)
        >>>
        >>> # To use Geometric in a network
        >>> class net(Cell):
        >>>     def __init__(self):
        >>>         super(net, self).__init__():
        >>>         self.g1 = msd.Geometric(0.5, dtype=mstype.int32)
        >>>         self.g2 = msd.Geometric(dtype=mstype.int32)
        >>>
        >>>     # Tthe following calls are valid in construct
        >>>     def construct(self, value, probs_b, probs_a):
        >>>
        >>>         # Similar calls can be made to other probability functions
        >>>         # by replacing 'prob' with the name of the function
        >>>         ans = self.g1.prob(value)
        >>>         # Evaluate with the respect to distribution b
        >>>         ans = self.g1.prob(value, probs_b)
        >>>
        >>>         # Probs must be passed in during function calls
        >>>         ans = self.g2.prob(value, probs_a)
        >>>
        >>>         # Functions 'sd', 'var', 'entropy' have the same usage as 'mean'
        >>>         # Will return 1.0
        >>>         ans = self.g1.mean()
        >>>         # Another possible usage
        >>>         ans = self.g1.mean(probs_b)
        >>>
        >>>         # Probs must be passed in during function calls
        >>>         ans = self.g2.mean(probs_a)
        >>>
        >>>         # Usage of 'kl_loss' and 'cross_entropy' are similar
        >>>         ans = self.g1.kl_loss('Geometric', probs_b)
        >>>         ans = self.g1.kl_loss('Geometric', probs_b, probs_a)
        >>>
        >>>         # Additional probs must be passed in
        >>>         ans = self.g2.kl_loss('Geometric', probs_b, probs_a)
        >>>
        >>>         # Sample
        >>>         ans = self.g1.sample()
        >>>         ans = self.g1.sample((2,3))
        >>>         ans = self.g1.sample((2,3), probs_b)
        >>>         ans = self.g2.sample((2,3), probs_a)
    """

    def __init__(self,
                 probs=None,
                 seed=0,
                 dtype=mstype.int32,
                 name="Geometric"):
        """
        Constructor of Geometric distribution.
        """
        param = dict(locals())
        valid_dtype = mstype.int_type + mstype.uint_type + mstype.float_type
        check_type(dtype, valid_dtype, type(self).__name__)
        super(Geometric, self).__init__(seed, dtype, name, param)
        self.parameter_type = mstype.float32
        if probs is not None:
            self._probs = cast_to_tensor(probs, self.parameter_type)
            check_prob(self._probs)
        else:
            self._probs = probs

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
        self.sqrt = P.Sqrt()
        self.uniform = C.uniform

    def extend_repr(self):
        if self.is_scalar_batch:
            str_info = f'probs = {self.probs}'
        else:
            str_info = f'batch_shape = {self._broadcast_shape}'
        return str_info

    @property
    def probs(self):
        """
        Returns the probability of success of the Bernoulli trail.
        """
        return self._probs

    def _check_param(self, probs1):
        """
        Check availablity of distribution specific args probs1.
        """
        if probs1 is not None:
            if self.context_mode == 0:
                self.checktensor(probs1, 'probs1')
            else:
                probs1 = self.checktensor(probs1, 'probs1')
            return self.cast(probs1, self.parameter_type)
        return self.probs if self.probs is not None else raise_none_error('probs1')

    def _mean(self, probs1=None):
        r"""
        .. math::
            MEAN(Geo) = \fratc{1 - probs1}{probs1}
        """
        probs1 = self._check_param(probs1)
        return (1. - probs1) / probs1

    def _mode(self, probs1=None):
        r"""
        .. math::
            MODE(Geo) = 0
        """
        probs1 = self._check_param(probs1)
        return self.fill(self.dtypeop(probs1), self.shape(probs1), 0.)

    def _var(self, probs1=None):
        r"""
        .. math::
            VAR(Geo) = \frac{1 - probs1}{probs1 ^ {2}}
        """
        probs1 = self._check_param(probs1)
        return (1.0 - probs1) / self.sq(probs1)

    def _entropy(self, probs1=None):
        r"""
        .. math::
            H(Geo) = \frac{-1 * probs0 \log_2 (1-probs0)\ - prob1 * \log_2 (1-probs1)\ }{probs1}
        """
        probs1 = self._check_param(probs1)
        probs0 = 1.0 - probs1
        return (-probs0 * self.log(probs0) - probs1 * self.log(probs1)) / probs1

    def _cross_entropy(self, dist, probs1_b, probs1=None):
        r"""
        Evaluate cross_entropy between Geometric distributions.

        Args:
            dist (str): type of the distributions. Should be "Geometric" in this case.
            probs1_b (Tensor): probability of success of distribution b.
            probs1_a (Tensor): probability of success of distribution a. Default: self.probs.
        """
        check_distribution_name(dist, 'Geometric')
        return self._entropy(probs1) + self._kl_loss(dist, probs1_b, probs1)

    def _prob(self, value, probs1=None):
        r"""
        pmf of Geometric distribution.

        Args:
            value (Tensor): a Tensor composed of only natural numbers.
            probs (Tensor): probability of success. Default: self.probs.

        .. math::
            pmf(k) = probs0 ^k * probs1 if k >= 0;
            pmf(k) = 0 if k < 0.
        """
        value = self._check_value(value, 'value')
        value = self.cast(value, mstype.float32)
        value = self.floor(value)
        probs1 = self._check_param(probs1)
        pmf = self.exp(self.log(1.0 - probs1) * value + self.log(probs1))
        zeros = self.fill(self.dtypeop(probs1), self.shape(pmf), 0.0)
        comp = self.less(value, zeros)
        return self.select(comp, zeros, pmf)

    def _cdf(self, value, probs1=None):
        r"""
        cdf of Geometric distribution.

        Args:
            value (Tensor): a Tensor composed of only natural numbers.
            probs (Tensor): probability of success. Default: self.probs.

        .. math::
            cdf(k) = 1 - probs0 ^ (k+1) if k >= 0;
            cdf(k) = 0 if k < 0.

        """
        value = self._check_value(value, 'value')
        value = self.cast(value, mstype.float32)
        value = self.floor(value)
        probs1 = self._check_param(probs1)
        probs0 = 1.0 - probs1
        cdf = 1.0 - self.pow(probs0, value + 1.0)
        zeros = self.fill(self.dtypeop(probs1), self.shape(cdf), 0.0)
        comp = self.less(value, zeros)
        return self.select(comp, zeros, cdf)

    def _kl_loss(self, dist, probs1_b, probs1=None):
        r"""
        Evaluate Geometric-Geometric kl divergence, i.e. KL(a||b).

        Args:
            dist (str): type of the distributions. Should be "Geometric" in this case.
            probs1_b (Tensor): probability of success of distribution b.
            probs1_a (Tensor): probability of success of distribution a. Default: self.probs.

        .. math::
            KL(a||b) = \log(\frac{probs1_a}{probs1_b}) + \frac{probs0_a}{probs1_a} * \log(\frac{probs0_a}{probs0_b})
        """
        check_distribution_name(dist, 'Geometric')
        probs1_b = self._check_value(probs1_b, 'probs1_b')
        probs1_b = self.cast(probs1_b, self.parameter_type)
        probs1_a = self._check_param(probs1)
        probs0_a = 1.0 - probs1_a
        probs0_b = 1.0 - probs1_b
        return self.log(probs1_a / probs1_b) + (probs0_a / probs1_a) * self.log(probs0_a / probs0_b)

    def _sample(self, shape=(), probs1=None):
        """
        Sampling.

        Args:
            shape (tuple): shape of the sample. Default: ().
            probs (Tensor): probability of success. Default: self.probs.

        Returns:
            Tensor, shape is shape + batch_shape.
        """
        shape = self.checktuple(shape, 'shape')
        probs1 = self._check_param(probs1)
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
