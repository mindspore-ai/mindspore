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
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.common import dtype as mstype
from .distribution import Distribution
from ._utils.utils import cast_to_tensor, check_greater, check_type, check_distribution_name,\
    set_param_type
from ._utils.custom_ops import exp_generic, log_generic


class Uniform(Distribution):
    """
    Example class: Uniform Distribution.

    Args:
        low (int, float, list, numpy.ndarray, Tensor, Parameter): The lower bound of the distribution.
        high (int, float, list, numpy.ndarray, Tensor, Parameter): The upper bound of the distribution.
        seed (int): The seed uses in sampling. The global seed is used if it is None. Default: None.
        dtype (mindspore.dtype): The type of the event samples. Default: mstype.float32.
        name (str): The name of the distribution. Default: 'Uniform'.

    Note:
        `low` must be stricly less than `high`.
        `dist_spec_args` are `high` and `low`.
        `dtype` must be float type because Uniform distributions are continuous.

    Examples:
        >>> # To initialize a Uniform distribution of the lower bound 0.0 and the higher bound 1.0.
        >>> import mindspore.nn.probability.distribution as msd
        >>> u = msd.Uniform(0.0, 1.0, dtype=mstype.float32)
        >>>
        >>> # The following creates two independent Uniform distributions.
        >>> u = msd.Uniform([0.0, 0.0], [1.0, 2.0], dtype=mstype.float32)
        >>>
        >>> # A Uniform distribution can be initilized without arguments.
        >>> # In this case, `high` and `low` must be passed in through arguments during function calls.
        >>> u = msd.Uniform(dtype=mstype.float32)
        >>>
        >>> # To use a Uniform distribution in a network.
        >>> class net(Cell):
        >>>     def __init__(self)
        >>>         super(net, self).__init__():
        >>>         self.u1 = msd.Uniform(0.0, 1.0, dtype=mstype.float32)
        >>>         self.u2 = msd.Uniform(dtype=mstype.float32)
        >>>
        >>>     # All the following calls in construct are valid.
        >>>     def construct(self, value, low_b, high_b, low_a, high_a):
        >>>
        >>>         # Private interfaces of probability functions corresponding to public interfaces, including
        >>>         # `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, and `log_survival`, have the same arguments.
        >>>         # Args:
        >>>         #     value (Tensor): the value to be evaluated.
        >>>         #     low (Tensor): the lower bound of distribution. Default: self.low.
        >>>         #     high (Tensor): the higher bound of distribution. Default: self.high.
        >>>
        >>>         # Examples of `prob`.
        >>>         # Similar calls can be made to other probability functions
        >>>         # by replacing 'prob' by the name of the function.
        >>>         ans = self.u1.prob(value)
        >>>         # Evaluate with respect to distribution b.
        >>>         ans = self.u1.prob(value, low_b, high_b)
        >>>         # `high` and `low` must be passed in during function calls.
        >>>         ans = self.u2.prob(value, low_a, high_a)
        >>>
        >>>
        >>>         # Functions `mean`, `sd`, `var`, and `entropy` have the same arguments.
        >>>         # Args:
        >>>         #     low (Tensor): the lower bound of distribution. Default: self.low.
        >>>         #     high (Tensor): the higher bound of distribution. Default: self.high.
        >>>
        >>>         # Examples of `mean`. `sd`, `var`, and `entropy` are similar.
        >>>         ans = self.u1.mean() # return 0.5
        >>>         ans = self.u1.mean(low_b, high_b) # return (low_b + high_b) / 2
        >>>         # `high` and `low` must be passed in during function calls.
        >>>         ans = self.u2.mean(low_a, high_a)
        >>>
        >>>         # Interfaces of 'kl_loss' and 'cross_entropy' are the same.
        >>>         # Args:
        >>>         #     dist (str): the type of the distributions. Should be "Uniform" in this case.
        >>>         #     low_b (Tensor): the lower bound of distribution b.
        >>>         #     high_b (Tensor): the upper bound of distribution b.
        >>>         #     low_a (Tensor): the lower bound of distribution a. Default: self.low.
        >>>         #     high_a (Tensor): the upper bound of distribution a. Default: self.high.
        >>>
        >>>         # Examples of `kl_loss`. `cross_entropy` is similar.
        >>>         ans = self.u1.kl_loss('Uniform', low_b, high_b)
        >>>         ans = self.u1.kl_loss('Uniform', low_b, high_b, low_a, high_a)
        >>>         # Additional `high` and `low` must be passed in.
        >>>         ans = self.u2.kl_loss('Uniform', low_b, high_b, low_a, high_a)
        >>>
        >>>
        >>>         # Examples of `sample`.
        >>>         # Args:
        >>>         #     shape (tuple): the shape of the sample. Default: ()
        >>>         #     low (Tensor): the lower bound of the distribution. Default: self.low.
        >>>         #     high (Tensor): the upper bound of the distribution. Default: self.high.
        >>>         ans = self.u1.sample()
        >>>         ans = self.u1.sample((2,3))
        >>>         ans = self.u1.sample((2,3), low_b, high_b)
        >>>         ans = self.u2.sample((2,3), low_a, high_a)
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
        valid_dtype = mstype.float_type
        check_type(dtype, valid_dtype, type(self).__name__)
        super(Uniform, self).__init__(seed, dtype, name, param)
        self.parameter_type = set_param_type(
            {'low': low, 'high': high}, self.dtype)
        if low is not None and high is not None:
            self._low = cast_to_tensor(low, self.parameter_type)
            self._high = cast_to_tensor(high, self.parameter_type)
            check_greater(self.low, self.high, "low value", "high value")
        else:
            self._low = low if low is None else cast_to_tensor(
                low, self.parameter_type)
            self._high = high if high is None else cast_to_tensor(
                high, self.parameter_type)

        self.default_parameters = [self.low, self.high]
        self.parameter_names = ['low', 'high']

        # ops needed for the class
        self.exp = exp_generic
        self.log = log_generic
        self.squeeze = P.Squeeze(0)
        self.cast = P.Cast()
        self.const = P.ScalarToArray()
        self.dtypeop = P.DType()
        self.fill = P.Fill()
        self.less = P.Less()
        self.lessequal = P.LessEqual()
        self.logicaland = P.LogicalAnd()
        self.select = P.Select()
        self.shape = P.Shape()
        self.sq = P.Square()
        self.sqrt = P.Sqrt()
        self.zeroslike = P.ZerosLike()
        self.uniform = C.uniform

        self.sametypeshape = P.SameTypeShape()

    def extend_repr(self):
        if self.is_scalar_batch:
            str_info = f'low = {self.low}, high = {self.high}'
        else:
            str_info = f'batch_shape = {self._broadcast_shape}'
        return str_info

    @property
    def low(self):
        """
        Return the lower bound of the distribution.
        """
        return self._low

    @property
    def high(self):
        """
        Return the upper bound of the distribution.
        """
        return self._high

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
        Evaluate cross entropy between Uniform distributoins.

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
        return self.select(comp, kl, self.log(self.zeroslike(kl)))

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
        l_zero = self.const(0.0)
        h_one = self.const(1.0)
        sample_uniform = self.uniform(sample_shape, l_zero, h_one, self.seed)
        sample = (high - low) * sample_uniform + low
        value = self.cast(sample, self.dtype)
        if origin_shape == ():
            value = self.squeeze(value)
        return value
