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
from ._utils.utils import cast_to_tensor, check_prob, check_type

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
        probs should be proper probabilities (0 <= p <= 1).
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
        valid_dtype = mstype.int_type + mstype.uint_type
        check_type(dtype, valid_dtype, "Geometric")
        super(Geometric, self).__init__(seed, dtype, name, param)
        if probs is not None:
            self._probs = cast_to_tensor(probs, hint_dtype=mstype.float32)
            check_prob(self._probs)
        else:
            self._probs = probs

        self.minval = np.finfo(np.float).tiny

        # ops needed for the class
        self.cast = P.Cast()
        self.const = P.ScalarToArray()
        self.dtypeop = P.DType()
        self.exp = P.Exp()
        self.fill = P.Fill()
        self.floor = P.Floor()
        self.issubclass = P.IsSubClass()
        self.less = P.Less()
        self.log = P.Log()
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
        Returns the probability for the outcome is 1.
        """
        return self._probs

    def _mean(self, probs1=None):
        r"""
        .. math::
            MEAN(Geo) = \fratc{1 - probs1}{probs1}
        """
        probs1 = self.probs if probs1 is None else probs1
        return (1. - probs1) / probs1

    def _mode(self, probs1=None):
        r"""
        .. math::
            MODE(Geo) = 0
        """
        probs1 = self.probs if probs1 is None else probs1
        return self.fill(self.dtypeop(probs1), self.shape(probs1), 0.)

    def _var(self, probs1=None):
        r"""
        .. math::
            VAR(Geo) = \frac{1 - probs1}{probs1 ^ {2}}
        """
        probs1 = self.probs if probs1 is None else probs1
        return (1.0 - probs1) / self.sq(probs1)

    def _entropy(self, probs=None):
        r"""
        .. math::
            H(Geo) = \frac{-1 * probs0 \log_2 (1-probs0)\ - prob1 * \log_2 (1-probs1)\ }{probs1}
        """
        probs1 = self.probs if probs is None else probs
        probs0 = 1.0 - probs1
        return (-probs0 * self.log(probs0) - probs1 * self.log(probs1)) / probs1

    def _cross_entropy(self, dist, probs1_b, probs1_a=None):
        r"""
        Evaluate cross_entropy between Geometric distributions.

        Args:
            dist (str): type of the distributions. Should be "Geometric" in this case.
            probs1_b (Tensor): probability of success of distribution b.
            probs1_a (Tensor): probability of success of distribution a. Default: self.probs.
        """
        if dist == 'Geometric':
            return self._entropy(probs=probs1_a) + self._kl_loss(dist, probs1_b, probs1_a)
        return None

    def _prob(self, value, probs=None):
        r"""
        pmf of Geometric distribution.

        Args:
            value (Tensor): a Tensor composed of only natural numbers.
            probs (Tensor): probability of success. Default: self.probs.

        .. math::
            pmf(k) = probs0 ^k * probs1 if k >= 0;
            pmf(k) = 0 if k < 0.
        """
        probs1 = self.probs if probs is None else probs
        dtype = self.dtypeop(value)
        if self.issubclass(dtype, mstype.int_):
            pass
        elif self.issubclass(dtype, mstype.float_):
            value = self.floor(value)
        else:
            return None
        pmf = self.exp(self.log(1.0 - probs1) * value + self.log(probs1))
        zeros = self.fill(self.dtypeop(probs1), self.shape(pmf), 0.0)
        comp = self.less(value, zeros)
        return self.select(comp, zeros, pmf)

    def _cdf(self, value, probs=None):
        r"""
        cdf of Geometric distribution.

        Args:
            value (Tensor): a Tensor composed of only natural numbers.
            probs (Tensor): probability of success. Default: self.probs.

        .. math::
            cdf(k) = 1 - probs0 ^ (k+1) if k >= 0;
            cdf(k) = 0 if k < 0.

        """
        probs1 = self.probs if probs is None else probs
        probs0 = 1.0 - probs1
        dtype = self.dtypeop(value)
        if self.issubclass(dtype, mstype.int_):
            pass
        elif self.issubclass(dtype, mstype.float_):
            value = self.floor(value)
        else:
            return None
        cdf = 1.0 - self.pow(probs0, value + 1.0)
        zeros = self.fill(self.dtypeop(probs1), self.shape(cdf), 0.0)
        comp = self.less(value, zeros)
        return self.select(comp, zeros, cdf)


    def _kl_loss(self, dist, probs1_b, probs1_a=None):
        r"""
        Evaluate Geometric-Geometric kl divergence, i.e. KL(a||b).

        Args:
            dist (str): type of the distributions. Should be "Geometric" in this case.
            probs1_b (Tensor): probability of success of distribution b.
            probs1_a (Tensor): probability of success of distribution a. Default: self.probs.

        .. math::
            KL(a||b) = \log(\frac{probs1_a}{probs1_b}) + \frac{probs0_a}{probs1_a} * \log(\frac{probs0_a}{probs0_b})
        """
        if dist == 'Geometric':
            probs1_a = self.probs if probs1_a is None else probs1_a
            probs0_a = 1.0 - probs1_a
            probs0_b = 1.0 - probs1_b
            return self.log(probs1_a / probs1_b) + (probs0_a / probs1_a) * self.log(probs0_a / probs0_b)
        return None

    def _sample(self, shape=(), probs=None):
        """
        Sampling.

        Args:
            shape (tuple): shape of the sample. Default: ().
            probs (Tensor): probability of success. Default: self.probs.

        Returns:
            Tensor, shape is shape + batch_shape.
        """
        probs = self.probs if probs is None else probs
        minval = self.const(self.minval)
        maxval = self.const(1.0)
        sample_uniform = self.uniform(shape + self.shape(probs), minval, maxval, self.seed)
        sample = self.floor(self.log(sample_uniform) / self.log(1.0 - probs))
        return self.cast(sample, self.dtype)
