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
"""Normal Distribution"""
import numpy as np
from mindspore.ops import operations as P
from .distribution import Distribution
from ._utils.utils import convert_to_batch, check_greater_equal_zero
from ...common import dtype as mstype
from ...context import get_context

class Normal(Distribution):
    """
    Example class: Normal distribution.

    Args:
        mean (int/float/list/numpy.ndarray/Tensor): mean of the Gaussian distribution
        standard deviation (int/float/list/numpy.ndarray/Tensor): vairance of the Gaussian distribution
        dtype (mindspore.dtype): type of the distribution

    Note:
        Standard deviation should be greater than zero.

    Examples:
        >>>    # To initialize a normal distribution of mean 3.0 and standard deviation 4.0
        >>>    n = nn.Normal(3.0, 4.0, dtype=dtype.float32)
        >>>    # The following create two independent normal distributions
        >>>    n = nn.Normal([3.0, 3.0], [4.0, 4.0], dtype=dtype.float32)
    """

    def __init__(self,
                 mean=None,
                 sd=None,
                 dtype=mstype.float32,
                 name="Normal"):
        """
        Constructor of normal distribution.
        """
        param = dict(locals())
        super(Normal, self).__init__(dtype, name, param)
        if  mean is not None and sd is not None:
            self._mean_value = convert_to_batch(mean, self._broadcast_shape, dtype)
            self._sd_value = convert_to_batch(sd, self._broadcast_shape, dtype)
            #check validity of standard deviation
            check_greater_equal_zero(self._sd_value, "Standard deviation")
        else:
            self._mean_value = mean
            self._sd_value = sd

        #ops needed for the class
        self.exp = P.Exp()
        self.add = P.TensorAdd()
        self.sq = P.Square()
        self.log = P.Log()
        self.sqrt = P.Sqrt()
        self.realdiv = P.RealDiv()
        self.expm1 = P.Expm1() if get_context('device_target') == 'Ascend' else self._expm1_by_step

    def _expm1_by_step(self, x):
        """
        Expm1 ops under GPU context.
        """
        return self.add(self.exp(x), -1)

    def _mean(self):
        """
        Mean of the distribution.
        """
        return self._mean_value

    def _sd(self):
        """
        Standard deviation of the distribution.
        """
        return self._sd_value

    def _log_likelihood(self, name, value, mean=None, sd=None):
        r"""
        Evaluate log probability.

        .. math::
            L(x) = -1* \fract{(x - \mu)^2}{2. * \sigma^2} - \log(\sqrt(2* \pi * \sigma^2))
        """
        mean = self._mean_value if mean is None else mean
        sd = self._sd_value if sd is None else sd
        unnormalized_log_prob = -1. * self.realdiv(self.sq(self.add(value, -1. * mean)),
                                                   2. * self.sq(sd))
        neg_normalization = -1. * self.log(self.sqrt(2. * np.pi * self.sq(sd)))
        return self.add(unnormalized_log_prob, neg_normalization)

    def _kl_loss(self, name, dist, mean, sd):
        r"""
        Evaluate Normal-Normal kl divergence, i.e. KL(a||b).

        Args:
            name (str): name of the funtion passed in from construct. Should always be "kl_loss".
            dist (str): type of the distributions. Should be "Normal" in this case.
            mean (Tensor): mean of distribution b.
            sd (Tensor): standard deviation distribution b.

        .. math::
            KL(a||b) = 0.5 * (\fract{MEAN(a)}{STD(b)} - \fract{MEAN(b)}{STD(b)}) ^ 2 +
                       0.5 * EXPM1(2 * (\log(STD(a)) - \log(STD(b))) - (\log(STD(a)) - \log(STD(b)))
        """
        if dist == 'Normal':
            diff_log_scale = self.add(self.log(self._sd_value), - self.log(sd))
            squared_diff = self.sq(self.add(self.realdiv(self._mean_value, sd), - self.realdiv(mean, sd)))
            return self.add(self.add(0.5 * squared_diff, 0.5 * self.expm1(2 * diff_log_scale)), - diff_log_scale)
        return None

    def extend_repr(self):
        str_info = 'mean={}, standard deviation={}'.format(self._mean_value, self._sd_value)
        return str_info
