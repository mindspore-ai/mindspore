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
from .distribution import Distribution
from ._utils.utils import cast_to_tensor, check_prob
from ...common import dtype as mstype

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
        >>> n = nn.Geometric(0.5, dtype=mstype.int32)
        >>>
        >>> # The following creates two independent Geometric distributions
        >>> n = nn.Geometric([0.5, 0.5], dtype=mstype.int32)
        >>>
        >>> # A Geometric distribution can be initilized without arguments
        >>> # In this case, probs must be passed in through construct.
        >>> n = nn.Geometric(dtype=mstype.int32)
        >>>
        >>> # To use Geometric distribution in a network
        >>> class net(Cell):
        >>>     def __init__(self):
        >>>         super(net, self).__init__():
        >>>         self.g1 = nn.Geometric(0.5, dtype=mstype.int32)
        >>>         self.g2 = nn.Geometric(dtype=mstype.int32)
        >>>
        >>>     # Tthe following calls are valid in construct
        >>>     def construct(self, value, probs_b, probs_a):
        >>>
        >>>         # Similar calls can be made to other probability functions
        >>>         # by replacing 'prob' with the name of the function
        >>>         ans = self.g1('prob', value)
        >>>         # Evaluate with the respect to distribution b
        >>>         ans = self.g1('prob', value, probs_b)
        >>>
        >>>         # Probs must be passed in through construct
        >>>         ans = self.g2('prob', value, probs_a)
        >>>
        >>>         # Functions 'sd', 'var', 'entropy' have the same usage with 'mean'
        >>>         # Will return [0.0]
        >>>         ans = self.g1('mean')
        >>>         # Will return mean_b
        >>>         ans = self.g1('mean', probs_b)
        >>>
        >>>         # Probs must be passed in through construct
        >>>         ans = self.g2('mean', probs_a)
        >>>
        >>>         # Usage of 'kl_loss' and 'cross_entropy' are similar
        >>>         ans = self.g1('kl_loss', 'Geometric', probs_b)
        >>>         ans = self.g1('kl_loss', 'Geometric', probs_b, probs_a)
        >>>
        >>>         # Additional probs must be passed in through construct
        >>>         ans = self.g2('kl_loss', 'Geometric', probs_b, probs_a)
        >>>
        >>>         # Sample Usage
        >>>         ans = self.g1('sample')
        >>>         ans = self.g1('sample', (2,3))
        >>>         ans = self.g1('sample', (2,3), probs_b)
        >>>         ans = self.g2('sample', (2,3), probs_a)
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
        super(Geometric, self).__init__(dtype, name, param)
        if probs is not None:
            self._probs = cast_to_tensor(probs, dtype=mstype.float32)
            check_prob(self._probs)
        else:
            self._probs = probs

        self.minval = np.finfo(np.float).tiny

        # ops needed for the class
        self.const = P.ScalarToArray()
        self.dtypeop = P.DType()
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
        self.uniform = P.UniformReal(seed=seed)

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

    def _mean(self, name='mean', probs1=None):
        r"""
        .. math::
            MEAN(Geo) = \fratc{1 - probs1}{probs1}
        """
        if name == 'mean':
            probs1 = self.probs if probs1 is None else probs1
            return (1. - probs1) / probs1
        return None

    def _mode(self, name='mode', probs1=None):
        r"""
        .. math::
            MODE(Geo) = 0
        """
        if name == 'mode':
            probs1 = self.probs if probs1 is None else probs1
            return self.fill(self.dtypeop(probs1), self.shape(probs1), 0.)
        return None

    def _var(self, name='var', probs1=None):
        r"""
        .. math::
            VAR(Geo) = \fract{1 - probs1}{probs1 ^ {2}}
        """
        if name in self._variance_functions:
            probs1 = self.probs if probs1 is None else probs1
            return (1.0 - probs1) / self.sq(probs1)
        return None

    def _entropy(self, name='entropy', probs=None):
        r"""
        .. math::
            H(Geo) = \fract{-1 * probs0 \log_2 (1-probs0)\ - prob1 * \log_2 (1-probs1)\ }{probs1}
        """
        if name == 'entropy':
            probs1 = self.probs if probs is None else probs
            probs0 = 1.0 - probs1
            return (-probs0 * self.log(probs0) - probs1 * self.log(probs1)) / probs1
        return None

    def _cross_entropy(self, name, dist, probs1_b, probs1_a=None):
        r"""
        Evaluate cross_entropy between Geometric distributions.

        Args:
            name (str): name of the funtion. Should always be "cross_entropy" when passed in from construct.
            dist (str): type of the distributions. Should be "Geometric" in this case.
            probs1_b (Tensor): probability of success of distribution b.
            probs1_a (Tensor): probability of success of distribution a. Default: self.probs.
        """
        if name == 'cross_entropy' and dist == 'Geometric':
            return self._entropy(probs=probs1_a) + self._kl_loss(name, dist, probs1_b, probs1_a)
        return None

    def _prob(self, name, value, probs=None):
        r"""
        pmf of Geometric distribution.

        Args:
            name (str): name of the function. Should be "prob" when passed in from construct.
            value (Tensor): a Tensor composed of only natural numbers.
            probs (Tensor): probability of success. Default: self.probs.

        .. math::
            pmf(k) = probs0 ^k * probs1 if k >= 0;
            pmf(k) = 0 if k < 0.
        """
        if name in self._prob_functions:
            probs1 = self.probs if probs is None else probs
            dtype = self.dtypeop(value)
            if self.issubclass(dtype, mstype.int_):
                pass
            elif self.issubclass(dtype, mstype.float_):
                value = self.floor(value)
            else:
                return None
            pmf = self.pow((1.0 - probs1), value) * probs1
            zeros = self.fill(self.dtypeop(probs1), self.shape(pmf), 0.0)
            comp = self.less(value, zeros)
            return self.select(comp, zeros, pmf)
        return None

    def _cdf(self, name, value, probs=None):
        r"""
        cdf of Geometric distribution.

        Args:
            name (str): name of the function.
            value (Tensor): a Tensor composed of only natural numbers.
            probs (Tensor): probability of success. Default: self.probs.

        .. math::
            cdf(k) = 1 - probs0 ^ (k+1) if k >= 0;
            cdf(k) = 0 if k < 0.

        """
        if name in self._cdf_survival_functions:
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
        return None

    def _kl_loss(self, name, dist, probs1_b, probs1_a=None):
        r"""
        Evaluate Geometric-Geometric kl divergence, i.e. KL(a||b).

        Args:
            name (str): name of the funtion.
            dist (str): type of the distributions. Should be "Geometric" in this case.
            probs1_b (Tensor): probability of success of distribution b.
            probs1_a (Tensor): probability of success of distribution a. Default: self.probs.

        .. math::
            KL(a||b) = \log(\fract{probs1_a}{probs1_b}) + \fract{probs0_a}{probs1_a} * \log(\fract{probs0_a}{probs0_b})
        """
        if name in self._divergence_functions and dist == 'Geometric':
            probs1_a = self.probs if probs1_a is None else probs1_a
            probs0_a = 1.0 - probs1_a
            probs0_b = 1.0 - probs1_b
            return self.log(probs1_a / probs1_b) + (probs0_a / probs1_a) * self.log(probs0_a / probs0_b)
        return None

    def _sample(self, name, shape=(), probs=None):
        """
        Sampling.

        Args:
            name (str): name of the function. Should always be 'sample' when passed in from construct.
            shape (tuple): shape of the sample. Default: ().
            probs (Tensor): probability of success. Default: self.probs.

        Returns:
            Tensor, shape is shape + batch_shape.
        """
        if name == 'sample':
            probs = self.probs if probs is None else probs
            minval = self.const(self.minval)
            maxval = self.const(1.0)
            sample_uniform = self.uniform(shape + self.shape(probs), minval, maxval)
            return self.floor(self.log(sample_uniform) / self.log(1.0 - probs))
        return None
