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
from ._utils.utils import cast_to_tensor, check_greater_zero, check_type, check_distribution_name,\
                          raise_none_error, common_dtype
from ._utils.custom_ops import exp_generic, expm1_generic, log_generic, erf_generic

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
        check_type(dtype, valid_dtype, type(self).__name__)
        super(Normal, self).__init__(seed, dtype, name, param)
        self.parameter_type = common_dtype(mean, 'mean', sd, 'sd', self.dtype)
        if  mean is not None and sd is not None:
            self._mean_value = cast_to_tensor(mean, self.parameter_type)
            self._sd_value = cast_to_tensor(sd, self.parameter_type)
            check_greater_zero(self._sd_value, "Standard deviation")
        else:
            self._mean_value = mean
            self._sd_value = sd

        #ops needed for the class
        self.exp = exp_generic
        self.expm1 = expm1_generic
        self.log = log_generic
        self.erf = erf_generic
        self.squeeze = P.Squeeze(0)
        self.cast = P.Cast()
        self.const = P.ScalarToArray()
        self.fill = P.Fill()
        self.shape = P.Shape()
        self.sq = P.Square()
        self.sqrt = P.Sqrt()
        self.zeroslike = P.ZerosLike()
        self.dtypeop = P.DType()
        self.sametypeshape = P.SameTypeShape()

    def extend_repr(self):
        if self.is_scalar_batch:
            str_info = f'mean = {self._mean_value}, standard deviation = {self._sd_value}'
        else:
            str_info = f'batch_shape = {self._broadcast_shape}'
        return str_info

    def _check_param(self, mean, sd):
        """
        Check availablity of distribution specific args mean and sd.
        """
        if mean is not None:
            if self.context_mode == 0:
                self.checktensor(mean, 'mean')
            else:
                mean = self.checktensor(mean, 'mean')
        else:
            mean = self._mean_value if self._mean_value is not None else raise_none_error('mean')
        if sd is not None:
            if self.context_mode == 0:
                self.checktensor(sd, 'sd')
            else:
                sd = self.checktensor(sd, 'sd')
        else:
            sd = self._sd_value if self._sd_value is not None else raise_none_error('sd')
        batch_shape = self.shape(mean + sd)
        mean = mean * self.fill(self.dtypeop(mean), batch_shape, 1.0)
        sd = sd * self.fill(self.dtypeop(sd), batch_shape, 1.0)
        self.sametypeshape(mean, sd)
        mean = self.cast(mean, self.parameter_type)
        sd = self.cast(sd, self.parameter_type)
        return mean, sd

    def _mean(self, mean=None, sd=None):
        """
        Mean of the distribution.
        """
        mean, sd = self._check_param(mean, sd)
        return mean

    def _mode(self, mean=None, sd=None):
        """
        Mode of the distribution.
        """
        mean, sd = self._check_param(mean, sd)
        return mean

    def _sd(self, mean=None, sd=None):
        """
        Standard deviation of the distribution.
        """
        mean, sd = self._check_param(mean, sd)
        return sd

    def _entropy(self, mean=None, sd=None):
        r"""
        Evaluate entropy.

        .. math::
            H(X) = \log(\sqrt(numpy.e * 2. * numpy.pi * \sq(\sigma)))
        """
        mean, sd = self._check_param(mean, sd)
        return self.log(self.sqrt(self.const(np.e * 2. * np.pi))) + self.log(sd)

    def _cross_entropy(self, dist, mean_b, sd_b, mean=None, sd=None):
        r"""
        Evaluate cross_entropy between normal distributions.

        Args:
            dist (str): type of the distributions. Should be "Normal" in this case.
            mean_b (Tensor): mean of distribution b.
            sd_b (Tensor): standard deviation distribution b.
            mean_a (Tensor): mean of distribution a. Default: self._mean_value.
            sd_a (Tensor): standard deviation distribution a. Default: self._sd_value.
        """
        check_distribution_name(dist, 'Normal')
        return self._entropy(mean, sd) + self._kl_loss(dist, mean_b, sd_b, mean, sd)

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
        value = self._check_value(value, 'value')
        value = self.cast(value, self.dtype)
        mean, sd = self._check_param(mean, sd)
        unnormalized_log_prob = -1. * (self.sq(value - mean)) / (2. * self.sq(sd))
        neg_normalization = -1. * self.log(self.const(2. * np.pi)) / 2. - self.log(sd)
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
        value = self._check_value(value, 'value')
        value = self.cast(value, self.dtype)
        mean, sd = self._check_param(mean, sd)
        sqrt2 = self.sqrt(self.const(2.0))
        adjusted = (value - mean) / (sd * sqrt2)
        return 0.5 * (1.0 + self.erf(adjusted))

    def _kl_loss(self, dist, mean_b, sd_b, mean=None, sd=None):
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
        check_distribution_name(dist, 'Normal')
        mean_b = self._check_value(mean_b, 'mean_b')
        sd_b = self._check_value(sd_b, 'sd_b')
        mean_b = self.cast(mean_b, self.parameter_type)
        sd_b = self.cast(sd_b, self.parameter_type)
        mean_a, sd_a = self._check_param(mean, sd)
        diff_log_scale = self.log(sd_a) - self.log(sd_b)
        squared_diff = self.sq(mean_a / sd_b - mean_b / sd_b)
        return 0.5 * squared_diff + 0.5 * self.expm1(2 * diff_log_scale) - diff_log_scale

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
        shape = self.checktuple(shape, 'shape')
        mean, sd = self._check_param(mean, sd)
        batch_shape = self.shape(mean + sd)
        origin_shape = shape + batch_shape
        if origin_shape == ():
            sample_shape = (1,)
        else:
            sample_shape = origin_shape
        sample_norm = C.normal(sample_shape, mean, sd, self.seed)
        value = self.cast(sample_norm, self.dtype)
        if origin_shape == ():
            value = self.squeeze(value)
        return value
