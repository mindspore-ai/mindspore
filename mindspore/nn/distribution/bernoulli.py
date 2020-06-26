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
from mindspore.ops import operations as P
from .distribution import Distribution
from ._utils.utils import cast_to_tensor, check_prob
from ...common import dtype as mstype

class Bernoulli(Distribution):
    """
    Example class: Bernoulli Distribution.

    Args:
        probs (int/float/list/numpy.ndarray/Tensor): probability of 1 as outcome.
        dtype (mindspore.dtype): type of the distribution, default to int32.

    Note:
        probs should be proper probabilities (0 <= p <= 1).

    Examples:
        >>>    # To initialize a Bernoulli distribution which has equal probability of getting 1 and 0
        >>>    b = nn.Bernoulli(0.5, dtype = dtype.int32)
        >>>    # The following create two independent Bernoulli distributions
        >>>    b = nn.Bernoulli([0.7, 0.2], dtype = dtype.int32)
    """

    def __init__(self,
                 probs=None,
                 dtype=mstype.int32,
                 name="Bernoulli"):
        """
        Constructor of Bernoulli distribution.
        """
        param = dict(locals())
        super(Bernoulli, self).__init__(dtype, name, param)
        if probs is not None:
            self._probs = cast_to_tensor(probs)
            # check if the input probability is valid
            check_prob(self._probs)
        else:
            self._probs = probs

        # ops needed for the class
        self.log = P.Log()
        self.add = P.TensorAdd()
        self.mul = P.Mul()
        self.sqrt = P.Sqrt()
        self.realdiv = P.RealDiv()


    def probs(self):
        """
        Returns the probability for the outcome is 1.
        """
        return self._probs

    def _mean(self):
        r"""
        .. math::
            MEAN(B) = probs1
        """

        return self._probs

    def _var(self):
        r"""
        .. math::
            VAR(B) = probs1 * probs0
        """
        probs0 = self.add(1, -1 * self._probs)
        return self.mul(probs0, self._probs)

    def _prob(self, name, value, probs=None):
        r"""
        pmf of Bernoulli distribution.

        Args:
            name (str): name of the function. Should be "prob" when passed in from construct.
            value (Tensor): a Tensor composed of only zeros and ones.
            probs (Tensor): probability of outcome is 1. Default to self._probs.

        .. math::
            pmf(k) = probs1 if k = 1;
            pmf(k) = probs0 if k = 0;
        """
        probs1 = self._probs if probs is None else probs
        probs0 = self.add(1, -1 * probs1)
        return self.add(self.mul(probs1, value),
                        self.mul(probs0, self.add(1, -1 * value)))

    def _kl_loss(self, name, dist, probs1_b):
        r"""
        Evaluate bernoulli-bernoulli kl divergence, i.e. KL(a||b).

        Args:
            name (str): name of the funtion. Should always be "kl_loss" when passed in from construct.
            dist (str): type of the distributions. Should be "Bernoulli" in this case.
            probs1_b (Tensor): probs1 of distribution b.

        .. math::
            KL(a||b) = probs1_a * \log(\fract{probs1_a}{probs1_b}) +
                       probs0_a * \log(\fract{probs0_a}{probs0_b})
        """
        if dist == 'Bernoulli':
            probs1_a = self._probs
            probs0_a = self.add(1, -1 * probs1_a)
            probs0_b = self.add(1, -1 * probs1_b)
            return self.add(probs1_a * self.log(self.realdiv(probs1_a, probs1_b)),
                            probs0_a * self.log(self.realdiv(probs0_a, probs0_b)))
        return None

    def extend_repr(self):
        str_info = 'probs={}'.format(self._probs)
        return str_info
