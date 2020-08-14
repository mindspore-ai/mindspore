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
from mindspore.common import dtype as mstype
from .distribution import Distribution
from ._utils.utils import convert_to_batch, check_greater_zero, check_type


class Normal(Distribution):
    """
    Normal distribution.

    Args:
        mean (int, float, list, numpy.ndarray, Tensor, Parameter): mean of the Normal distribution.
        sd (int, float, list, numpy.ndarray, Tensor, Parameter): stddev of the Normal distribution.
        seed (int): seed to use in sampling. Default: 0.
        dtype (mindspore.dtype): type of the distribution. Default: mstype.float32.
        name (str): name of the distribution. Default: Normal.

    Note:
        Standard deviation should be greater than zero.
        Dist_spec_args are mean and sd.

    Examples:
        >>> # To initialize a Normal distribution of mean 3.0 and standard deviation 4.0
        >>> import mindspore.nn.probability.distribution as msd
        >>> n = msd.Normal(3.0, 4.0, dtype=mstype.float32)
        >>>
        >>> # The following creates two independent Normal distributions
        >>> n = msd.Normal([3.0, 3.0], [4.0, 4.0], dtype=mstype.float32)
        >>>
        >>> # A Normal distribution can be initilize without arguments
        >>> # In this case, mean and sd must be passed in through args.
        >>> n = msd.Normal(dtype=mstype.float32)
        >>>
        >>> # To use Normal in a network
        >>> class net(Cell):
        >>>     def __init__(self):
        >>>         super(net, self).__init__():
        >>>         self.n1 = msd.Nomral(0.0, 1.0, dtype=mstype.float32)
        >>>         self.n2 = msd.Normal(dtype=mstype.float32)
        >>>
        >>>     # The following calls are valid in construct
        >>>     def construct(self, value, mean_b, sd_b, mean_a, sd_a):
        >>>
        >>>         # Similar calls can be made to other probability functions
        >>>         # by replacing 'prob' with the name of the function
        >>>         ans = self.n1.prob(value)
        >>>         # Evaluate with the respect to distribution b
        >>>         ans = self.n1.prob(value, mean_b, sd_b)
        >>>
        >>>         # mean and sd must be passed in during function calls
        >>>         ans = self.n2.prob(value, mean_a, sd_a)
        >>>
        >>>         # Functions 'sd', 'var', 'entropy' have the same usage as 'mean'
        >>>         # will return [0.0]
        >>>         ans = self.n1.mean()
        >>>         # will return mean_b
        >>>         ans = self.n1.mean(mean_b, sd_b)
        >>>
        >>>         # mean and sd must be passed during function calls
        >>>         ans = self.n2.mean(mean_a, sd_a)
        >>>
        >>>         # Usage of 'kl_loss' and 'cross_entropy' are similar
        >>>         ans = self.n1.kl_loss('Normal', mean_b, sd_b)
        >>>         ans = self.n1.kl_loss('Normal', mean_b, sd_b, mean_a, sd_a)
        >>>
        >>>         # Additional mean and sd must be passed
        >>>         ans = self.n2.kl_loss('Normal', mean_b, sd_b, mean_a, sd_a)
        >>>
        >>>         # Sample
        >>>         ans = self.n1.sample()
        >>>         ans = self.n1.sample((2,3))
        >>>         ans = self.n1.sample((2,3), mean_b, sd_b)
        >>>         ans = self.n2.sample((2,3), mean_a, sd_a)
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
        valid_dtype = mstype.float_type
        check_type(dtype, valid_dtype, "Normal")
        super(Normal, self).__init__(seed, dtype, name, param)
        if  mean is not None and sd is not None:
            self._mean_value = convert_to_batch(mean, self.broadcast_shape, dtype)
            self._sd_value = convert_to_batch(sd, self.broadcast_shape, dtype)
            check_greater_zero(self._sd_value, "Standard deviation")
        else:
            self._mean_value = mean
            self._sd_value = sd


        #ops needed for the class
        self.const = P.ScalarToArray()
        self.erf = P.Erf()
        self.exp = P.Exp()
        self.expm1 = self._expm1_by_step
        self.fill = P.Fill()
        self.log = P.Log()
        self.shape = P.Shape()
        self.sq = P.Square()
        self.sqrt = P.Sqrt()
        self.zeroslike = P.ZerosLike()

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

    def _mean(self, mean=None, sd=None):
        """
        Mean of the distribution.
        """
        mean = self._mean_value if mean is None or sd is None else mean
        return mean

    def _mode(self, mean=None, sd=None):
        """
        Mode of the distribution.
        """
        mean = self._mean_value if mean is None or sd is None else mean
        return mean

    def _sd(self, mean=None, sd=None):
        """
        Standard deviation of the distribution.
        """
        sd = self._sd_value if mean is None or sd is None else sd
        return sd

    def _entropy(self, sd=None):
        r"""
        Evaluate entropy.

        .. math::
            H(X) = \log(\sqrt(numpy.e * 2. * numpy.pi * \sq(\sigma)))
        """
        sd = self._sd_value if sd is None else sd
        return self.log(self.sqrt(self.const(np.e * 2. * np.pi))) + self.log(sd)

    def _cross_entropy(self, dist, mean_b, sd_b, mean_a=None, sd_a=None):
        r"""
        Evaluate cross_entropy between normal distributions.

        Args:
            dist (str): type of the distributions. Should be "Normal" in this case.
            mean_b (Tensor): mean of distribution b.
            sd_b (Tensor): standard deviation distribution b.
            mean_a (Tensor): mean of distribution a. Default: self._mean_value.
            sd_a (Tensor): standard deviation distribution a. Default: self._sd_value.
        """
        if dist == 'Normal':
            return self._entropy(sd=sd_a) + self._kl_loss(dist, mean_b, sd_b, mean_a, sd_a)
        return None

    def _log_prob(self, value, mean=None, sd=None):
        r"""
        Evaluate log probability.

        Args:
            value (Tensor): value to be evaluated.
            mean (Tensor): mean of the distribution. Default: self._mean_value.
            sd (Tensor): standard deviation the distribution. Default: self._sd_value.

        .. math::
            L(x) = -1* \frac{(x - \mu)^2}{2. * \sigma^2} - \log(\sqrt(2* \pi * \sigma^2))
        """
        mean = self._mean_value if mean is None else mean
        sd = self._sd_value if sd is None else sd
        unnormalized_log_prob = -1. * (self.sq(value - mean)) / (2. * self.sq(sd))
        neg_normalization = -1. * self.log(self.sqrt(self.const(2. * np.pi))) - self.log(sd)
        return unnormalized_log_prob + neg_normalization

    def _cdf(self, value, mean=None, sd=None):
        r"""
        Evaluate cdf of given value.

        Args:
            value (Tensor): value to be evaluated.
            mean (Tensor): mean of the distribution. Default: self._mean_value.
            sd (Tensor): standard deviation the distribution. Default: self._sd_value.

        .. math::
            cdf(x) = 0.5 * (1+ Erf((x - \mu) / ( \sigma * \sqrt(2))))
        """
        mean = self._mean_value if mean is None else mean
        sd = self._sd_value if sd is None else sd
        sqrt2 = self.sqrt(self.const(2.0))
        adjusted = (value - mean) / (sd * sqrt2)
        return 0.5 * (1.0 + self.erf(adjusted))

    def _kl_loss(self, dist, mean_b, sd_b, mean_a=None, sd_a=None):
        r"""
        Evaluate Normal-Normal kl divergence, i.e. KL(a||b).

        Args:
            dist (str): type of the distributions. Should be "Normal" in this case.
            mean_b (Tensor): mean of distribution b.
            sd_b (Tensor): standard deviation distribution b.
            mean_a (Tensor): mean of distribution a. Default: self._mean_value.
            sd_a (Tensor): standard deviation distribution a. Default: self._sd_value.

        .. math::
            KL(a||b) = 0.5 * (\frac{MEAN(a)}{STD(b)} - \frac{MEAN(b)}{STD(b)}) ^ 2 +
                       0.5 * EXPM1(2 * (\log(STD(a)) - \log(STD(b))) - (\log(STD(a)) - \log(STD(b)))
        """
        if dist == 'Normal':
            mean_a = self._mean_value if mean_a is None else mean_a
            sd_a = self._sd_value if sd_a is None else sd_a
            diff_log_scale = self.log(sd_a) - self.log(sd_b)
            squared_diff = self.sq(mean_a / sd_b - mean_b / sd_b)
            return 0.5 * squared_diff + 0.5 * self.expm1(2 * diff_log_scale) - diff_log_scale
        return None

    def _sample(self, shape=(), mean=None, sd=None):
        """
        Sampling.

        Args:
            shape (tuple): shape of the sample. Default: ().
            mean (Tensor): mean of the samples. Default: self._mean_value.
            sd (Tensor): standard deviation of the samples. Default: self._sd_value.

        Returns:
            Tensor, shape is shape + batch_shape.
        """
        mean = self._mean_value if mean is None else mean
        sd = self._sd_value if sd is None else sd
        batch_shape = self.shape(self.zeroslike(mean) + self.zeroslike(sd))
        sample_shape = shape + batch_shape
        sample_norm = C.normal(sample_shape, mean, sd, self.seed)
        return sample_norm
