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
"""Uniform Distribution"""
import numpy as np
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore._checkparam import Validator
from mindspore.common import dtype as mstype
from .distribution import Distribution
from ._utils.utils import check_greater, check_distribution_name
from ._utils.custom_ops import exp_generic, log_generic


class Uniform(Distribution):
    r"""
    Uniform Distribution.
    A Uniform distributio is a continuous distribution with the range :math:`[a, b]`
    and the probability density function:

    .. math::
        f(x, a, b) = 1 / (b - a),

    where :math:`a, b` are the lower and upper bound respectively.

    Args:
        low (int, float, list, numpy.ndarray, Tensor): The lower bound of the distribution. Default: None.
        high (int, float, list, numpy.ndarray, Tensor): The upper bound of the distribution. Default: None.
        seed (int): The seed uses in sampling. The global seed is used if it is None. Default: None.
        dtype (mindspore.dtype): The type of the event samples. Default: mstype.float32.
        name (str): The name of the distribution. Default: 'Uniform'.

    Note:
        `low` must be strictly less than `high`.
        `dist_spec_args` are `high` and `low`.
        `dtype` must be float type because Uniform distributions are continuous.

    Raises:
        ValueError: When high <= low.
        TypeError: When the input `dtype` is not a subclass of float.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> import mindspore.nn.probability.distribution as msd
        >>> from mindspore import Tensor
        >>> # To initialize a Uniform distribution of the lower bound 0.0 and the higher bound 1.0.
        >>> u1 = msd.Uniform(0.0, 1.0, dtype=mindspore.float32)
        >>> # A Uniform distribution can be initialized without arguments.
        >>> # In this case, `high` and `low` must be passed in through arguments during function calls.
        >>> u2 = msd.Uniform(dtype=mindspore.float32)
        >>>
        >>> # Here are some tensors used below for testing
        >>> value = Tensor([0.5, 0.8], dtype=mindspore.float32)
        >>> low_a = Tensor([0., 0.], dtype=mindspore.float32)
        >>> high_a = Tensor([2.0, 4.0], dtype=mindspore.float32)
        >>> low_b = Tensor([-1.5], dtype=mindspore.float32)
        >>> high_b = Tensor([2.5, 5.], dtype=mindspore.float32)
        >>> # Private interfaces of probability functions corresponding to public interfaces, including
        >>> # `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, and `log_survival`, have the same arguments.
        >>> # Args:
        >>> #     value (Tensor): the value to be evaluated.
        >>> #     low (Tensor): the lower bound of the distribution. Default: self.low.
        >>> #     high (Tensor): the higher bound of the distribution. Default: self.high.
        >>> # Examples of `prob`.
        >>> # Similar calls can be made to other probability functions
        >>> # by replacing 'prob' by the name of the function.
        >>> ans = u1.prob(value)
        >>> print(ans.shape)
        (2,)
        >>> # Evaluate with respect to distribution b.
        >>> ans = u1.prob(value, low_b, high_b)
        >>> print(ans.shape)
        (2,)
        >>> # `high` and `low` must be passed in during function calls.
        >>> ans = u2.prob(value, low_a, high_a)
        >>> print(ans.shape)
        (2,)
        >>> # Functions `mean`, `sd`, `var`, and `entropy` have the same arguments.
        >>> # Args:
        >>> #     low (Tensor): the lower bound of the distribution. Default: self.low.
        >>> #     high (Tensor): the higher bound of the distribution. Default: self.high.
        >>> # Examples of `mean`. `sd`, `var`, and `entropy` are similar.
        >>> ans = u1.mean() # return 0.5
        >>> print(ans.shape)
        ()
        >>> ans = u1.mean(low_b, high_b) # return (low_b + high_b) / 2
        >>> print(ans.shape)
        (2,)
        >>> # `high` and `low` must be passed in during function calls.
        >>> ans = u2.mean(low_a, high_a)
        >>> print(ans.shape)
        (2,)
        >>> # Interfaces of 'kl_loss' and 'cross_entropy' are the same.
        >>> # Args:
        >>> #     dist (str): the type of the distributions. Should be "Uniform" in this case.
        >>> #     low_b (Tensor): the lower bound of distribution b.
        >>> #     high_b (Tensor): the upper bound of distribution b.
        >>> #     low_a (Tensor): the lower bound of distribution a. Default: self.low.
        >>> #     high_a (Tensor): the upper bound of distribution a. Default: self.high.
        >>> # Examples of `kl_loss`. `cross_entropy` is similar.
        >>> ans = u1.kl_loss('Uniform', low_b, high_b)
        >>> print(ans.shape)
        (2,)
        >>> ans = u1.kl_loss('Uniform', low_b, high_b, low_a, high_a)
        >>> print(ans.shape)
        (2,)
        >>> # Additional `high` and `low` must be passed in.
        >>> ans = u2.kl_loss('Uniform', low_b, high_b, low_a, high_a)
        >>> print(ans.shape)
        (2,)
        >>> # Examples of `sample`.
        >>> # Args:
        >>> #     shape (tuple): the shape of the sample. Default: ()
        >>> #     low (Tensor): the lower bound of the distribution. Default: self.low.
        >>> #     high (Tensor): the upper bound of the distribution. Default: self.high.
        >>> ans = u1.sample()
        >>> print(ans.shape)
        ()
        >>> ans = u1.sample((2,3))
        >>> print(ans.shape)
        (2, 3)
        >>> ans = u1.sample((2,3), low_b, high_b)
        >>> print(ans.shape)
        (2, 3, 2)
        >>> ans = u2.sample((2,3), low_a, high_a)
        >>> print(ans.shape)
        (2, 3, 2)
    """

    def __init__(self,
                 low=None,
                 high=None,
                 seed=None,
                 dtype=mstype.float32,
                 name="Uniform"):
        """
        Constructor of Uniform distribution.
        """
        param = dict(locals())
        param['param_dict'] = {'low': low, 'high': high}
        valid_dtype = mstype.float_type
        Validator.check_type_name(
            "dtype", dtype, valid_dtype, type(self).__name__)
        super(Uniform, self).__init__(seed, dtype, name, param)

        self._low = self._add_parameter(low, 'low')
        self._high = self._add_parameter(high, 'high')
        if self.low is not None and self.high is not None:
            check_greater(self.low, self.high, 'low', 'high')

        # ops needed for the class
        self.exp = exp_generic
        self.log = log_generic
        self.squeeze = P.Squeeze(0)
        self.cast = P.Cast()
        self.const = P.ScalarToTensor()
        self.dtypeop = P.DType()
        self.fill = P.Fill()
        self.less = P.Less()
        self.lessequal = P.LessEqual()
        self.logicaland = P.LogicalAnd()
        self.select = P.Select()
        self.shape = P.Shape()
        self.sq = P.Square()
        self.zeroslike = P.ZerosLike()
        self.uniform = C.uniform

    def extend_repr(self):
        """Display instance object as string."""
        if self.is_scalar_batch:
            s = 'low = {}, high = {}'.format(self.low, self.high)
        else:
            s = 'batch_shape = {}'.format(self._broadcast_shape)
        return s

    @property
    def low(self):
        """
        Return the lower bound of the distribution after casting to dtype.

        Output:
            Tensor, the lower bound of the distribution.
        """
        return self._low

    @property
    def high(self):
        """
        Return the upper bound of the distribution after casting to dtype.

        Output:
            Tensor, the upper bound of the distribution.
        """
        return self._high

    def _get_dist_type(self):
        return "Uniform"

    def _get_dist_args(self, low=None, high=None):
        if low is not None:
            self.checktensor(low, 'low')
        else:
            low = self.low
        if high is not None:
            self.checktensor(high, 'high')
        else:
            high = self.high
        return low, high

    def _range(self, low=None, high=None):
        r"""
        Return the range of the distribution.

        .. math::
            range(U) = high -low
        """
        low, high = self._check_param_type(low, high)
        return high - low

    def _mean(self, low=None, high=None):
        r"""
        .. math::
            MEAN(U) = \frac{low + high}{2}.
        """
        low, high = self._check_param_type(low, high)
        return (low + high) / 2.

    def _var(self, low=None, high=None):
        r"""
        .. math::
            VAR(U) = \frac{(high -low) ^ 2}{12}.
        """
        low, high = self._check_param_type(low, high)
        return self.sq(high - low) / 12.0

    def _entropy(self, low=None, high=None):
        r"""
        .. math::
            H(U) = \log(high - low).
        """
        low, high = self._check_param_type(low, high)
        return self.log(high - low)

    def _cross_entropy(self, dist, low_b, high_b, low=None, high=None):
        """
        Evaluate cross entropy between Uniform distributions.

        Args:
            dist (str): The type of the distributions. Should be "Uniform" in this case.
            low_b (Tensor): The lower bound of distribution b.
            high_b (Tensor): The upper bound of distribution b.
            low_a (Tensor): The lower bound of distribution a. Default: self.low.
            high_a (Tensor): The upper bound of distribution a. Default: self.high.
        """
        check_distribution_name(dist, 'Uniform')
        return self._entropy(low, high) + self._kl_loss(dist, low_b, high_b, low, high)

    def _prob(self, value, low=None, high=None):
        r"""
        pdf of Uniform distribution.

        Args:
            value (Tensor): The value to be evaluated.
            low (Tensor): The lower bound of the distribution. Default: self.low.
            high (Tensor): The upper bound of the distribution. Default: self.high.

        .. math::
            pdf(x) = 0 if x < low;
            pdf(x) = \frac{1.0}{high -low} if low <= x <= high;
            pdf(x) = 0 if x > high;
        """
        value = self._check_value(value, 'value')
        value = self.cast(value, self.dtype)
        low, high = self._check_param_type(low, high)
        neg_ones = self.fill(self.dtype, self.shape(value), -1.0)
        prob = self.exp(neg_ones * self.log(high - low))
        broadcast_shape = self.shape(prob)
        zeros = self.fill(self.dtypeop(prob), broadcast_shape, 0.0)
        comp_lo = self.less(value, low)
        comp_hi = self.lessequal(value, high)
        less_than_low = self.select(comp_lo, zeros, prob)
        return self.select(comp_hi, less_than_low, zeros)

    def _kl_loss(self, dist, low_b, high_b, low=None, high=None):
        """
        Evaluate uniform-uniform KL divergence, i.e. KL(a||b).

        Args:
            dist (str): The type of the distributions. Should be "Uniform" in this case.
            low_b (Tensor): The lower bound of distribution b.
            high_b (Tensor): The upper bound of distribution b.
            low_a (Tensor): The lower bound of distribution a. Default: self.low.
            high_a (Tensor): The upper bound of distribution a. Default: self.high.
        """
        check_distribution_name(dist, 'Uniform')
        low_b = self._check_value(low_b, 'low_b')
        low_b = self.cast(low_b, self.parameter_type)
        high_b = self._check_value(high_b, 'high_b')
        high_b = self.cast(high_b, self.parameter_type)
        low_a, high_a = self._check_param_type(low, high)
        kl = self.log(high_b - low_b) - self.log(high_a - low_a)
        comp = self.logicaland(self.lessequal(
            low_b, low_a), self.lessequal(high_a, high_b))
        inf = self.fill(self.dtypeop(kl), self.shape(kl), np.inf)
        return self.select(comp, kl, inf)

    def _cdf(self, value, low=None, high=None):
        r"""
        The cumulative distribution function of Uniform distribution.

        Args:
            value (Tensor): The value to be evaluated.
            low (Tensor): The lower bound of the distribution. Default: self.low.
            high (Tensor): The upper bound of the distribution. Default: self.high.

        .. math::
            cdf(x) = 0 if x < low;
            cdf(x) = \frac{x - low}{high -low} if low <= x <= high;
            cdf(x) = 1 if x > high;
        """
        value = self._check_value(value, 'value')
        value = self.cast(value, self.dtype)
        low, high = self._check_param_type(low, high)
        prob = (value - low) / (high - low)
        broadcast_shape = self.shape(prob)
        zeros = self.fill(self.dtypeop(prob), broadcast_shape, 0.0)
        ones = self.fill(self.dtypeop(prob), broadcast_shape, 1.0)
        comp_lo = self.less(value, low)
        comp_hi = self.less(value, high)
        less_than_low = self.select(comp_lo, zeros, prob)
        return self.select(comp_hi, less_than_low, ones)

    def _sample(self, shape=(), low=None, high=None):
        """
        Sampling.

        Args:
            shape (tuple): The shape of the sample. Default: ().
            low (Tensor): The lower bound of the distribution. Default: self.low.
            high (Tensor): The upper bound of the distribution. Default: self.high.

        Returns:
            Tensor, with the shape being shape + batch_shape.
        """
        shape = self.checktuple(shape, 'shape')
        low, high = self._check_param_type(low, high)
        broadcast_shape = self.shape(low + high)
        origin_shape = shape + broadcast_shape
        if origin_shape == ():
            sample_shape = (1,)
        else:
            sample_shape = origin_shape
        l_zero = self.const(0.0, mstype.float32)
        h_one = self.const(1.0, mstype.float32)
        sample_uniform = self.uniform(sample_shape, l_zero, h_one, self.seed)
        sample = (high - low) * sample_uniform + low
        value = self.cast(sample, self.dtype)
        if origin_shape == ():
            value = self.squeeze(value)
        return value
