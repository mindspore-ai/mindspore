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
from mindspore import _checkparam as Validator
from mindspore.common import dtype as mstype
from mindspore.common import Tensor
from .distribution import Distribution
from ._utils.utils import check_greater_zero, check_distribution_name


class Normal(Distribution):
    r"""
    Normal distribution.
    A Normal distribution is a continuous distribution with the range :math:`(-\inf, \inf)`
    and the probability density function:

    .. math::
        f(x, \mu, \sigma) = 1 / \sigma\sqrt{2\pi} \exp(-(x - \mu)^2 / 2\sigma^2).

    where :math:`\mu, \sigma` are the mean and
    the standard deviation of the normal distribution respectively.

    Args:
        mean (int, float, list, numpy.ndarray, Tensor): The mean of the Normal distribution. Default: ``None`` .
        sd (int, float, list, numpy.ndarray, Tensor): The standard deviation of the Normal distribution.
            Default: ``None`` .
        seed (int): The seed used in sampling. The global seed is used if it is None. Default: ``None`` .
        dtype (mindspore.dtype): The type of the event samples. Default: ``mstype.float32`` .
        name (str): The name of the distribution. Default: ``'Normal'`` .

    Note:
        `sd` must be greater than zero.
        `dist_spec_args` are `mean` and `sd`.
        `dtype` must be a float type because Normal distributions are continuous.

    Raises:
        ValueError: When sd <= 0.
        TypeError: When the input `dtype` is not a subclass of float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> import mindspore.nn.probability.distribution as msd
        >>> from mindspore import Tensor
        >>> # To initialize a Normal distribution of the mean 3.0 and the standard deviation 4.0.
        >>> n1 = msd.Normal(3.0, 4.0, dtype=mindspore.float32)
        >>> # A Normal distribution can be initialized without arguments.
        >>> # In this case, `mean` and `sd` must be passed in through arguments.
        >>> n2 = msd.Normal(dtype=mindspore.float32)
        >>> # Here are some tensors used below for testing
        >>> value = Tensor([1.0, 2.0, 3.0], dtype=mindspore.float32)
        >>> mean_a = Tensor([2.0], dtype=mindspore.float32)
        >>> sd_a = Tensor([2.0, 2.0, 2.0], dtype=mindspore.float32)
        >>> mean_b = Tensor([1.0], dtype=mindspore.float32)
        >>> sd_b = Tensor([1.0, 1.5, 2.0], dtype=mindspore.float32)
        >>> # Private interfaces of probability functions corresponding to public interfaces, including
        >>> # `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, and `log_survival`,
        >>> # have the same arguments as follows.
        >>> # Args:
        >>> #     value (Tensor): the value to be evaluated.
        >>> #     mean (Tensor): the mean of the distribution. Default: self._mean_value.
        >>> #     sd (Tensor): the standard deviation of the distribution. Default: self._sd_value.
        >>> # Examples of `prob`.
        >>> # Similar calls can be made to other probability functions
        >>> # by replacing 'prob' by the name of the function
        >>> ans = n1.prob(value)
        >>> print(ans.shape)
        (3,)
        >>> # Evaluate with respect to the distribution b.
        >>> ans = n1.prob(value, mean_b, sd_b)
        >>> print(ans.shape)
        (3,)
        >>> # `mean` and `sd` must be passed in during function calls
        >>> ans = n2.prob(value, mean_a, sd_a)
        >>> print(ans.shape)
        (3,)
        >>> # Functions `mean`, `sd`, `var`, and `entropy` have the same arguments.
        >>> # Args:
        >>> #     mean (Tensor): the mean of the distribution. Default: self._mean_value.
        >>> #     sd (Tensor): the standard deviation of the distribution. Default: self._sd_value.
        >>> # Example of `mean`. `sd`, `var`, and `entropy` are similar.
        >>> ans = n1.mean() # return 0.0
        >>> print(ans.shape)
        ()
        >>> ans = n1.mean(mean_b, sd_b) # return mean_b
        >>> print(ans.shape)
        (3,)
        >>> # `mean` and `sd` must be passed in during function calls.
        >>> ans = n2.mean(mean_a, sd_a)
        >>> print(ans.shape)
        (3,)
        >>> # Interfaces of 'kl_loss' and 'cross_entropy' are the same:
        >>> # Args:
        >>> #     dist (str): the type of the distributions. Only "Normal" is supported.
        >>> #     mean_b (Tensor): the mean of distribution b.
        >>> #     sd_b (Tensor): the standard deviation of distribution b.
        >>> #     mean_a (Tensor): the mean of distribution a. Default: self._mean_value.
        >>> #     sd_a (Tensor): the standard deviation of distribution a. Default: self._sd_value.
        >>> # Examples of `kl_loss`. `cross_entropy` is similar.
        >>> ans = n1.kl_loss('Normal', mean_b, sd_b)
        >>> print(ans.shape)
        (3,)
        >>> ans = n1.kl_loss('Normal', mean_b, sd_b, mean_a, sd_a)
        >>> print(ans.shape)
        (3,)
        >>> # Additional `mean` and `sd` must be passed in.
        >>> ans = n2.kl_loss('Normal', mean_b, sd_b, mean_a, sd_a)
        >>> print(ans.shape)
        (3,)
        >>> # Examples of `sample`.
        >>> # Args:
        >>> #     shape (tuple): the shape of the sample. Default: ()
        >>> #     mean (Tensor): the mean of the distribution. Default: self._mean_value.
        >>> #     sd (Tensor): the standard deviation of the distribution. Default: self._sd_value.
        >>> ans = n1.sample()
        >>> print(ans.shape)
        ()
        >>> ans = n1.sample((2,3))
        >>> print(ans.shape)
        (2, 3)
        >>> ans = n1.sample((2,3), mean_b, sd_b)
        >>> print(ans.shape)
        (2, 3, 3)
        >>> ans = n2.sample((2,3), mean_a, sd_a)
        >>> print(ans.shape)
        (2, 3, 3)
    """

    def __init__(self,
                 mean=None,
                 sd=None,
                 seed=None,
                 dtype=mstype.float32,
                 name="Normal"):
        """
        Constructor of Normal.
        """
        param = dict(locals())
        param['param_dict'] = {'mean': mean, 'sd': sd}
        valid_dtype = mstype.float_type
        Validator.check_type_name(
            "dtype", dtype, valid_dtype, type(self).__name__)
        super(Normal, self).__init__(seed, dtype, name, param)

        self._mean_value = self._add_parameter(mean, 'mean')
        self._sd_value = self._add_parameter(sd, 'sd')
        if self._sd_value is not None:
            check_greater_zero(self._sd_value, "Standard deviation")

        # ops needed for the class
        self.exp = self.exp_base
        self.log = self.log_base
        self.expm1 = P.Expm1()
        self.erf = P.Erf()
        self.squeeze = P.Squeeze(0)
        self.cast = P.Cast()
        self.const = P.ScalarToTensor()
        self.shape = P.Shape()
        self.sq = P.Square()
        self.sqrt = P.Sqrt()
        self.coff = Tensor(-0.5 * np.log(2. * np.pi), dtype=dtype)

    def extend_repr(self):
        """Display instance object as string."""
        if self.is_scalar_batch:
            s = 'mean = {}, standard deviation = {}'.format(
                self._mean_value, self._sd_value)
        else:
            s = 'batch_shape = {}'.format(self._broadcast_shape)
        return s

    def _get_dist_type(self):
        return "Normal"

    def _get_dist_args(self, mean=None, sd=None):
        if mean is not None:
            self.checktensor(mean, 'mean')
        else:
            mean = self._mean_value
        if sd is not None:
            self.checktensor(sd, 'sd')
        else:
            sd = self._sd_value
        return mean, sd

    def _mean(self, mean=None, sd=None):
        """
        The mean of the distribution.
        """
        mean, sd = self._check_param_type(mean, sd)
        return mean

    def _mode(self, mean=None, sd=None):
        """
        The mode of the distribution.
        """
        mean, sd = self._check_param_type(mean, sd)
        return mean

    def _sd(self, mean=None, sd=None):
        """
        The standard deviation of the distribution.
        """
        mean, sd = self._check_param_type(mean, sd)
        return sd

    def _entropy(self, mean=None, sd=None):
        r"""
        Evaluate entropy.

        .. math::
            H(X) = \log(\sqrt(numpy.e * 2. * numpy.pi * \sq(\sigma)))
        """
        mean, sd = self._check_param_type(mean, sd)
        return self.log(self.sqrt(self.const(np.e * 2. * np.pi, mstype.float32))) + self.log(sd)

    def _cross_entropy(self, dist, mean_b, sd_b, mean=None, sd=None):
        r"""
        Evaluate cross entropy between normal distributions.

        Args:
            dist (str): Type of the distributions. Should be "Normal" in this case.
            mean_b (Tensor): Mean of distribution b.
            sd_b (Tensor): Standard deviation distribution b.
            mean_a (Tensor): Mean of distribution a. Default: self._mean_value.
            sd_a (Tensor): Standard deviation distribution a. Default: self._sd_value.
        """
        check_distribution_name(dist, 'Normal')
        return self._entropy(mean, sd) + self._kl_loss(dist, mean_b, sd_b, mean, sd)

    def _log_prob(self, value, mean=None, sd=None):
        r"""
        Evaluate log probability.

        Args:
            value (Tensor): The value to be evaluated.
            mean (Tensor): The mean of the distribution. Default: self._mean_value.
            sd (Tensor): The standard deviation the distribution. Default: self._sd_value.

        .. math::
            L(x) = -1* \frac{(x - \mu)^2}{2. * \sigma^2} - \log(\sqrt(2* \pi * \sigma^2))
        """
        value = self._check_value(value, 'value')
        value = self.cast(value, self.dtype)
        mean, sd = self._check_param_type(mean, sd)
        unnormalized_log_prob = -0.5 * (self.sq((value - mean) / sd))
        neg_normalization = self.coff - self.log(sd)
        return unnormalized_log_prob + neg_normalization

    def _cdf(self, value, mean=None, sd=None):
        r"""
        Evaluate the cumulative distribution function on the given value.

        Args:
            value (Tensor): The value to be evaluated.
            mean (Tensor): The mean of the distribution. Default: self._mean_value.
            sd (Tensor): The standard deviation the distribution. Default: self._sd_value.

        .. math::
            cdf(x) = 0.5 * (1+ Erf((x - \mu) / ( \sigma * \sqrt(2))))
        """
        value = self._check_value(value, 'value')
        value = self.cast(value, self.dtype)
        mean, sd = self._check_param_type(mean, sd)
        sqrt2 = self.sqrt(self.const(2.0, mstype.float32))
        adjusted = (value - mean) / (sd * sqrt2)
        return 0.5 * (1.0 + self.erf(adjusted))

    def _kl_loss(self, dist, mean_b, sd_b, mean=None, sd=None):
        r"""
        Evaluate Normal-Normal KL divergence, i.e. KL(a||b).

        Args:
            dist (str): The type of the distributions. Should be "Normal" in this case.
            mean_b (Tensor): The mean of distribution b.
            sd_b (Tensor): The standard deviation distribution b.
            mean_a (Tensor): The mean of distribution a. Default: self._mean_value.
            sd_a (Tensor): The standard deviation distribution a. Default: self._sd_value.

        .. math::
            KL(a||b) = 0.5 * (\frac{MEAN(a)}{STD(b)} - \frac{MEAN(b)}{STD(b)}) ^ 2 +
                       0.5 * EXPM1(2 * (\log(STD(a)) - \log(STD(b))) - (\log(STD(a)) - \log(STD(b)))
        """
        check_distribution_name(dist, 'Normal')
        mean_b = self._check_value(mean_b, 'mean_b')
        sd_b = self._check_value(sd_b, 'sd_b')
        mean_b = self.cast(mean_b, self.parameter_type)
        sd_b = self.cast(sd_b, self.parameter_type)
        mean_a, sd_a = self._check_param_type(mean, sd)
        diff_log_scale = self.log(sd_a) - self.log(sd_b)
        squared_diff = self.sq(mean_a / sd_b - mean_b / sd_b)
        return 0.5 * squared_diff + 0.5 * self.expm1(2 * diff_log_scale) - diff_log_scale

    def _sample(self, shape=(), mean=None, sd=None):
        """
        Sampling.

        Args:
            shape (tuple): The shape of the sample. Default: ().
            mean (Tensor): The mean of the samples. Default: self._mean_value.
            sd (Tensor): The standard deviation of the samples. Default: self._sd_value.

        Returns:
            Tensor, with the shape being shape + batch_shape.
        """
        shape = self.checktuple(shape, 'shape')
        mean, sd = self._check_param_type(mean, sd)
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
