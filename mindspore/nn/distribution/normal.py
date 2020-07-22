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
from mindspore.ops import composite as C
from .distribution import Distribution
from ._utils.utils import convert_to_batch, check_greater_equal_zero
from ...common import dtype as mstype
from ...context import get_context

class Normal(Distribution):
    """
    Normal distribution.

    Args:
        mean (int, float, list, numpy.ndarray, Tensor, Parameter): mean of the Gaussian distribution.
        sd (int, float, list, numpy.ndarray, Tensor, Parameter): stddev of the Gaussian distribution.
        seed (int): seed to use in sampling. Default: 0.
        dtype (mindspore.dtype): type of the distribution. Default: mstype.float32.
        name (str): name of the distribution. Default: Normal.


    Note:
        Standard deviation should be greater than zero.
        Dist_spec_args are mean and sd.

    Examples:
        >>> # To initialize a normal distribution of mean 3.0 and standard deviation 4.0
        >>> n = nn.Normal(3.0, 4.0, dtype=mstype.float32)
        >>>
        >>> # The following create two independent normal distributions
        >>> n = nn.Normal([3.0, 3.0], [4.0, 4.0], dtype=mstype.float32)
        >>>
        >>> # A normal distribution can be initilize without arguments
        >>> # In this case, mean and sd must be passed in through construct.
        >>> n = nn.Normal(dtype=mstype.float32)
        >>>
        >>> # To use normal in a network
        >>> class net(Cell):
        >>>     def __init__(self):
        >>>         super(net, self).__init__():
        >>>         self.n1 = nn.Normal(0.0, 1.0, dtype=mstype.float32)
        >>>         self.n2 = nn.Normal(dtype=mstype.float32)
        >>>
        >>>     # All the following calls in construct are valid
        >>>     def construct(self, value, mean_b, sd_b, mean_a, sd_a):
        >>>
        >>>         # Similar to calls can be made to other probability functions
        >>>         # by replacing 'prob' with the name of the function
        >>>         ans = self.n1('prob', value)
        >>>         # Evaluate with the respect to distribution b
        >>>         ans = self.n1('prob', value, mean_b, sd_b)
        >>>
        >>>         # Additional mean and sd must be passed in through construct
        >>>         ans = self.n2('prob', value, mean_a, sd_a)
        >>>
        >>>         # Functions 'sd', 'var', 'entropy' have the same usage with 'mean'
        >>>         # Will return [0.0]
        >>>         ans = self.n1('mean')
        >>>         # Will return mean_b
        >>>         ans = self.n1('mean', mean_b, sd_b)
        >>>
        >>>         # Additional mean and sd must be passed in through construct
        >>>         ans = self.n2('mean', mean_a, sd_a)
        >>>
        >>>         # Usage of 'kl_loss' and 'cross_entropy' are similar
        >>>         ans = self.n1('kl_loss', 'Normal', mean_b, sd_b)
        >>>         ans = self.n1('kl_loss', 'Normal', mean_b, sd_b, mean_a, sd_a)
        >>>
        >>>         # Additional mean and sd must be passed in through construct
        >>>         ans = self.n2('kl_loss', 'Normal', mean_b, sd_b, mean_a, sd_a)
        >>>
        >>>         # Sample Usage
        >>>         ans = self.n1('sample')
        >>>         ans = self.n1('sample', (2,3))
        >>>         ans = self.n1('sample', (2,3), mean_b, sd_b)
        >>>         ans = self.n2('sample', (2,3), mean_a, sd_a)
    """

    def __init__(self,
                 mean=None,
                 sd=None,
                 seed=0,
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
            check_greater_equal_zero(self._sd_value, "Standard deviation")
        else:
            self._mean_value = mean
            self._sd_value = sd
        self.seed = seed

        #ops needed for the class
        self.exp = P.Exp()
        self.expm1 = P.Expm1() if get_context('device_target') == 'Ascend' else self._expm1_by_step
        self.log = P.Log()
        self.shape = P.Shape()
        self.sq = P.Square()
        self.log = P.Log()
        self.sqrt = P.Sqrt()
        self.realdiv = P.RealDiv()
        self.expm1 = P.Expm1() if get_context('device_target') == 'Ascend' else self._expm1_by_step
        self.shape = P.Shape()
        self.zeroslike = P.ZerosLike()
        self.const = P.ScalarToArray()
        self.erf = P.Erf()
        self.fill = P.Fill()

    def extend_repr(self):
        if self.is_scalar_batch:
            str_info = f'mean = {self._mean_value}, standard deviation = {self._sd_value}'
        else:
            str_info = f'batch_shape = {self._broadcast_shape}'
        return str_info

    def _expm1_by_step(self, x):
        """
        Expm1 ops under GPU context.
        """
        return self.exp(x) - 1.0

    def _mean(self, name='mean', mean=None, sd=None):
        """
        Mean of the distribution.
        """
        if name == 'mean':
            mean = self._mean_value if mean is None or sd is None else mean
            return mean
        return None

    def _mode(self, name='mode', mean=None, sd=None):
        """
        Mode of the distribution.
        """
        if name == 'mode':
            mean = self._mean_value if mean is None or sd is None else mean
            return mean
        return None

    def _sd(self, name='sd', mean=None, sd=None):
        """
        Standard deviation of the distribution.
        """
        if name in self._variance_functions:
            sd = self._sd_value if mean is None or sd is None else sd
            return sd
        return None

    def _entropy(self, name='entropy', sd=None):
        r"""
        Evaluate entropy.

        .. math::
            H(X) = \log(\sqrt(numpy.e * 2. * numpy.pi * \sq(\sigma)))
        """
        if name == 'entropy':
            sd = self._sd_value if sd is None else sd
            return self.log(self.sqrt(np.e * 2. * np.pi * self.sq(sd)))
        return None

    def _cross_entropy(self, name, dist, mean_b, sd_b, mean_a=None, sd_a=None):
        r"""
        Evaluate cross_entropy between normal distributions.

        Args:
            name (str): name of the funtion passed in from construct. Should always be "cross_entropy".
            dist (str): type of the distributions. Should be "Normal" in this case.
            mean_b (Tensor): mean of distribution b.
            sd_b (Tensor): standard deviation distribution b.
            mean_a (Tensor): mean of distribution a. Default: self._mean_value.
            sd_a (Tensor): standard deviation distribution a. Default: self._sd_value.
        """
        if name == 'cross_entropy' and dist == 'Normal':
            return self._entropy(sd=sd_a) + self._kl_loss(name, dist, mean_b, sd_b, mean_a, sd_a)
        return None

    def _log_prob(self, name, value, mean=None, sd=None):
        r"""
        Evaluate log probability.

        Args:
            name (str): name of the funtion passed in from construct.
            value (Tensor): value to be evaluated.
            mean (Tensor): mean of the distribution. Default: self._mean_value.
            sd (Tensor): standard deviation the distribution. Default: self._sd_value.

        .. math::
            L(x) = -1* \fract{(x - \mu)^2}{2. * \sigma^2} - \log(\sqrt(2* \pi * \sigma^2))
        """
        if name in self._prob_functions:
            mean = self._mean_value if mean is None else mean
            sd = self._sd_value if sd is None else sd
            unnormalized_log_prob = -1. * (self.sq(value - mean)) / (2. * self.sq(sd))
            neg_normalization = -1. * self.log(self.sqrt(2. * np.pi * self.sq(sd)))
            return unnormalized_log_prob + neg_normalization
        return None

    def _cdf(self, name, value, mean=None, sd=None):
        r"""
        Evaluate cdf of given value.

        Args:
            name (str): name of the funtion passed in from construct. Should always be "cdf".
            value (Tensor): value to be evaluated.
            mean (Tensor): mean of the distribution. Default: self._mean_value.
            sd (Tensor): standard deviation the distribution. Default: self._sd_value.

        .. math::
            cdf(x) = 0.5 * (1+ Erf((x - \mu) / ( \sigma * \sqrt(2))))
        """
        if name in self._cdf_survival_functions:
            mean = self._mean_value if mean is None else mean
            sd = self._sd_value if sd is None else sd
            sqrt2 = self.sqrt(self.fill(mstype.float32, self.shape(sd), 2.0))
            adjusted = (value - mean) / self.mul(sd, sqrt2)
            return 0.5 * (1.0 + self.erf(adjusted))
        return None

    def _kl_loss(self, name, dist, mean_b, sd_b, mean_a=None, sd_a=None):
        r"""
        Evaluate Normal-Normal kl divergence, i.e. KL(a||b).

        Args:
            name (str): name of the funtion passed in from construct.
            dist (str): type of the distributions. Should be "Normal" in this case.
            mean_b (Tensor): mean of distribution b.
            sd_b (Tensor): standard deviation distribution b.
            mean_a (Tensor): mean of distribution a. Default: self._mean_value.
            sd_a (Tensor): standard deviation distribution a. Default: self._sd_value.

        .. math::
            KL(a||b) = 0.5 * (\fract{MEAN(a)}{STD(b)} - \fract{MEAN(b)}{STD(b)}) ^ 2 +
                       0.5 * EXPM1(2 * (\log(STD(a)) - \log(STD(b))) - (\log(STD(a)) - \log(STD(b)))
        """
        if name in self._divergence_functions and dist == 'Normal':
            mean_a = self._mean_value if mean_a is None else mean_a
            sd_a = self._sd_value if sd_a is None else sd_a
            diff_log_scale = self.log(sd_a) - self.log(sd_b)
            squared_diff = self.sq(mean_a / sd_b - mean_b / sd_b)
            return 0.5 * squared_diff + 0.5 * self.expm1(2 * diff_log_scale) - diff_log_scale
        return None

    def _sample(self, name, shape=(), mean=None, sd=None):
        """
        Sampling.

        Args:
            name (str): name of the function. Should always be 'sample' when passed in from construct.
            shape (tuple): shape of the sample. Default: ().
            mean (Tensor): mean of the samples. Default: self._mean_value.
            sd (Tensor): standard deviation of the samples. Default: self._sd_value.

        Returns:
            Tensor, shape is shape + batch_shape.
        """
        if name == 'sample':
            mean = self._mean_value if mean is None else mean
            sd = self._sd_value if sd is None else sd
            batch_shape = self.shape(self.add(self.zeroslike(mean), self.zeroslike(sd)))
            sample_shape = shape + batch_shape
            mean_zero = self.const(0.0)
            sd_one = self.const(1.0)
            sample_norm = C.normal(sample_shape, mean_zero, sd_one, self.seed)
            sample = self.add(mean, self.mul(sample_norm, sd))
            return sample
        return None
