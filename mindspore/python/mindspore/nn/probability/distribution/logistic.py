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
"""Logistic Distribution"""
import numpy as np
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore import _checkparam as Validator
from mindspore.common import dtype as mstype
from .distribution import Distribution
from ._utils.utils import check_greater_zero
from ._utils.custom_ops import exp_generic, log_generic


class Logistic(Distribution):
    r"""
    Logistic distribution.
    A Logistic distributio is a continuous distribution with the range :math:`(-\inf, \inf)`
    and the probability density function:

    .. math::
        f(x, a, b) = 1 / b \exp(\exp(-(x - a) / b) - x).

    where :math:`a, b` are loc and scale parameter respectively.

    Args:
        loc (float, list, numpy.ndarray, Tensor): The location of the Logistic distribution. Default: ``None`` .
        scale (float, list, numpy.ndarray, Tensor): The scale of the Logistic distribution. Default: ``None`` .
        seed (int): The seed used in sampling. The global seed is used if it is None. Default: ``None`` .
        dtype (mindspore.dtype): The type of the event samples. Default: ``mstype.float32`` .
        name (str): The name of the distribution. Default: ``'Logistic'`` .

    Note:
        `scale` must be greater than zero.
        `dist_spec_args` are `loc` and `scale`.
        `dtype` must be a float type because Logistic distributions are continuous.

    Raises:
        ValueError: When scale <= 0.
        TypeError: When the input `dtype` is not a subclass of float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> import mindspore.nn.probability.distribution as msd
        >>> from mindspore import Tensor
        >>> # To initialize a Logistic distribution of loc 3.0 and scale 4.0.
        >>> l1 = msd.Logistic(3.0, 4.0, dtype=mindspore.float32)
        >>> # A Logistic distribution can be initialized without arguments.
        >>> # In this case, `loc` and `scale` must be passed in through arguments.
        >>> l2 = msd.Logistic(dtype=mindspore.float32)
        >>>
        >>> # Here are some tensors used below for testing
        >>> value = Tensor([1.0, 2.0, 3.0], dtype=mindspore.float32)
        >>> loc_a = Tensor([2.0], dtype=mindspore.float32)
        >>> scale_a = Tensor([2.0, 2.0, 2.0], dtype=mindspore.float32)
        >>> loc_b = Tensor([1.0], dtype=mindspore.float32)
        >>> scale_b = Tensor([1.0, 1.5, 2.0], dtype=mindspore.float32)
        >>>
        >>> # Private interfaces of probability functions corresponding to public interfaces, including
        >>> # `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, and `log_survival`,
        >>> # have the same arguments as follows.
        >>> # Args:
        >>> #     value (Tensor): the value to be evaluated.
        >>> #     loc (Tensor): the location of the distribution. Default: self.loc.
        >>> #     scale (Tensor): the scale of the distribution. Default: self.scale.
        >>> # Examples of `prob`.
        >>> # Similar calls can be made to other probability functions
        >>> # by replacing 'prob' by the name of the function
        >>> ans = l1.prob(value)
        >>> print(ans.shape)
        (3,)
        >>> # Evaluate with respect to distribution b.
        >>> ans = l1.prob(value, loc_b, scale_b)
        >>> print(ans.shape)
        (3,)
        >>> # `loc` and `scale` must be passed in during function calls
        >>> ans = l1.prob(value, loc_a, scale_a)
        >>> print(ans.shape)
        (3,)
        >>> # Functions `mean`, `mode`, `sd`, `var`, and `entropy` have the same arguments.
        >>> # Args:
        >>> #     loc (Tensor): the location of the distribution. Default: self.loc.
        >>> #     scale (Tensor): the scale of the distribution. Default: self.scale.
        >>> # Example of `mean`. `mode`, `sd`, `var`, and `entropy` are similar.
        >>> ans = l1.mean()
        >>> print(ans.shape)
        ()
        >>> ans = l1.mean(loc_b, scale_b)
        >>> print(ans.shape)
        (3,)
        >>> # `loc` and `scale` must be passed in during function calls.
        >>> ans = l1.mean(loc_a, scale_a)
        >>> print(ans.shape)
        (3,)
        >>> # Examples of `sample`.
        >>> # Args:
        >>> #     shape (tuple): the shape of the sample. Default: ()
        >>> #     loc (Tensor): the location of the distribution. Default: self.loc.
        >>> #     scale (Tensor): the scale of the distribution. Default: self.scale.
        >>> ans = l1.sample()
        >>> print(ans.shape)
        ()
        >>> ans = l1.sample((2,3))
        >>> print(ans.shape)
        (2, 3)
        >>> ans = l1.sample((2,3), loc_b, scale_b)
        >>> print(ans.shape)
        (2, 3, 3)
        >>> ans = l1.sample((2,3), loc_a, scale_a)
        >>> print(ans.shape)
        (2, 3, 3)
    """

    def __init__(self,
                 loc=None,
                 scale=None,
                 seed=None,
                 dtype=mstype.float32,
                 name="Logistic"):
        """
        Constructor of Logistic.
        """
        param = dict(locals())
        param['param_dict'] = {'loc': loc, 'scale': scale}
        valid_dtype = mstype.float_type
        Validator.check_type_name(
            "dtype", dtype, valid_dtype, type(self).__name__)
        super(Logistic, self).__init__(seed, dtype, name, param)

        self._loc = self._add_parameter(loc, 'loc')
        self._scale = self._add_parameter(scale, 'scale')
        if self._scale is not None:
            check_greater_zero(self._scale, "scale")

        # ops needed for the class
        self.cast = P.Cast()
        self.consttensor = P.ScalarToTensor()
        self.dtypeop = P.DType()
        self.exp = exp_generic
        self.expm1 = P.Expm1()
        self.less = P.Less()
        self.log = log_generic
        self.log1p = P.Log1p()
        self.logicalor = P.LogicalOr()
        self.erf = P.Erf()
        self.greater = P.Greater()
        self.sigmoid = P.Sigmoid()
        self.squeeze = P.Squeeze(0)
        self.select = P.Select()
        self.shape = P.Shape()
        self.softplus = self._softplus
        self.sqrt = P.Sqrt()
        self.uniform = C.uniform
        self.neg = P.Neg()

        self.threshold = np.log(np.finfo(np.float32).eps) + 1.
        self.tiny = np.finfo(np.float).tiny
        self.sd_const = np.pi / np.sqrt(3)

    def _softplus(self, x):
        too_small = self.less(x, self.threshold)
        too_large = self.greater(x, -self.threshold)
        too_small_value = self.exp(x)
        too_large_value = x
        too_small_or_too_large = self.logicalor(too_small, too_large)
        ones = F.fill(self.dtypeop(x), self.shape(x), 1.0)
        x = self.select(too_small_or_too_large, ones, x)
        y = self.log(self.exp(x) + 1.0)
        return self.select(too_small, too_small_value, self.select(too_large, too_large_value, y))

    def extend_repr(self):
        """Display instance object as string."""
        if self.is_scalar_batch:
            s = 'location = {}, scale = {}'.format(self._loc, self._scale)
        else:
            s = 'batch_shape = {}'.format(self._broadcast_shape)
        return s

    @property
    def loc(self):
        """
        Return the location of the distribution after casting to dtype.

        Output:
            Tensor, the loc parameter of the distribution.
        """
        return self._loc

    @property
    def scale(self):
        """
        Return the scale of the distribution after casting to dtype.

        Output:
            Tensor, the scale parameter of the distribution.
        """
        return self._scale

    def _get_dist_type(self):
        return "Logistic"

    def _get_dist_args(self, loc=None, scale=None):
        if loc is None:
            loc = self.loc
        else:
            self.checktensor(loc, 'loc')
        if scale is None:
            scale = self.scale
        else:
            self.checktensor(scale, 'scale')
        return loc, scale

    def _mean(self, loc=None, scale=None):
        """
        The mean of the distribution.
        """
        loc, scale = self._check_param_type(loc, scale)
        return loc

    def _mode(self, loc=None, scale=None):
        """
        The mode of the distribution.
        """
        loc, scale = self._check_param_type(loc, scale)
        return loc

    def _sd(self, loc=None, scale=None):
        """
        The standard deviation of the distribution.
        """
        _, scale = self._check_param_type(loc, scale)
        return scale * self.consttensor(self.sd_const, self.dtypeop(scale))

    def _entropy(self, loc=None, scale=None):
        r"""
        Evaluate entropy.

        .. math::
            H(X) = \log(scale) + 2.
        """
        loc, scale = self._check_param_type(loc, scale)
        return self.log(scale) + 2.

    def _log_prob(self, value, loc=None, scale=None):
        r"""
        Evaluate log probability.

        Args:
            value (Tensor): The value to be evaluated.
            loc (Tensor): The location of the distribution. Default: self.loc.
            scale (Tensor): The scale of the distribution. Default: self.scale.

        .. math::
            z = (x - \mu) / \sigma
            L(x) = -z * -2. * softplus(-z) - \log(\sigma)
        """
        value = self._check_value(value, 'value')
        value = self.cast(value, self.dtype)
        loc, scale = self._check_param_type(loc, scale)
        z = (value - loc) / scale
        return -z - 2. * self.softplus(-z) - self.log(scale)

    def _cdf(self, value, loc=None, scale=None):
        r"""
        Evaluate the cumulative distribution function on the given value.

        Args:
            value (Tensor): The value to be evaluated.
            loc (Tensor): The location of the distribution. Default: self.loc.
            scale (Tensor): The scale the distribution. Default: self.scale.

        .. math::
            cdf(x) = sigmoid((x - loc) / scale)
        """
        value = self._check_value(value, 'value')
        value = self.cast(value, self.dtype)
        loc, scale = self._check_param_type(loc, scale)
        z = (value - loc) / scale
        return self.sigmoid(z)

    def _log_cdf(self, value, loc=None, scale=None):
        r"""
        Evaluate the log cumulative distribution function on the given value.

        Args:
            value (Tensor): The value to be evaluated.
            loc (Tensor): The location of the distribution. Default: self.loc.
            scale (Tensor): The scale the distribution. Default: self.scale.

        .. math::
            log_cdf(x) = -softplus(-(x - loc) / scale)
        """
        value = self._check_value(value, 'value')
        value = self.cast(value, self.dtype)
        loc, scale = self._check_param_type(loc, scale)
        z = (value - loc) / scale
        return (-1) * self.softplus(-z)

    def _survival_function(self, value, loc=None, scale=None):
        r"""
        Evaluate the survival function on the given value.

        Args:
            value (Tensor): The value to be evaluated.
            loc (Tensor): The location of the distribution. Default: self.loc.
            scale (Tensor): The scale the distribution. Default: self.scale.

        .. math::
            survival(x) = sigmoid(-(x - loc) / scale)
        """
        value = self._check_value(value, 'value')
        value = self.cast(value, self.dtype)
        loc, scale = self._check_param_type(loc, scale)
        z = (value - loc) / scale
        return self.sigmoid(-z)

    def _log_survival(self, value, loc=None, scale=None):
        r"""
        Evaluate the log survival function on the given value.

        Args:
            value (Tensor): The value to be evaluated.
            loc (Tensor): The location of the distribution. Default: self.loc.
            scale (Tensor): The scale the distribution. Default: self.scale.

        .. math::
            survival(x) = -softplus((x - loc) / scale)
        """
        value = self._check_value(value, 'value')
        value = self.cast(value, self.dtype)
        loc, scale = self._check_param_type(loc, scale)
        z = (value - loc) / scale
        return (-1) * self.softplus(z)

    def _sample(self, shape=(), loc=None, scale=None):
        """
        Sampling.

        Args:
            shape (tuple): The shape of the sample. Default: ().
            loc (Tensor): The location of the samples. Default: self.loc.
            scale (Tensor): The scale of the samples. Default: self.scale.

        Returns:
            Tensor, with the shape being shape + batch_shape.
        """
        shape = self.checktuple(shape, 'shape')
        loc, scale = self._check_param_type(loc, scale)
        batch_shape = self.shape(loc + scale)
        origin_shape = shape + batch_shape
        if origin_shape == ():
            sample_shape = (1,)
        else:
            sample_shape = origin_shape
        l_zero = self.consttensor(self.tiny, mstype.float32)
        h_one = self.consttensor(1.0, mstype.float32)
        sample_uniform = self.uniform(sample_shape, l_zero, h_one, self.seed)
        sample = self.log(sample_uniform) - self.log1p(self.neg(sample_uniform))
        sample = sample * scale + loc
        value = self.cast(sample, self.dtype)
        if origin_shape == ():
            value = self.squeeze(value)
        return value
