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
"""Gumbel Distribution"""
import numpy as np
from mindspore.ops import operations as P
from mindspore._checkparam import Validator
from mindspore.common import dtype as mstype
import mindspore.nn as nn
import mindspore.nn.probability.bijector as msb
import mindspore.nn.probability.distribution as msd
from .transformed_distribution import TransformedDistribution
from ._utils.utils import check_distribution_name
from ._utils.custom_ops import exp_generic, log_generic


class Gumbel(TransformedDistribution):
    r"""
    Gumbel distribution.
    A Gumbel distributio is a continuous distribution with the range of all real numbers
    and the probability density function:

    .. math::
        f(x, a, b) = 1 / b \exp(\exp(-(x - a) / b) - x),

    where :math:`a, b` are loc and scale parameter respectively.

    Args:
        loc (int, float, list, numpy.ndarray, Tensor): The location of Gumbel distribution.
        scale (int, float, list, numpy.ndarray, Tensor): The scale of Gumbel distribution.
        seed (int): the seed used in sampling. The global seed is used if it is None. Default: 0.
        dtype (mindspore.dtype): type of the distribution. Default: mstype.float32.
        name (str): the name of the distribution. Default: 'Gumbel'.

    Note:
        `scale` must be greater than zero.
        `dist_spec_args` are `loc` and `scale`.
        `dtype` must be a float type because Gumbel distributions are continuous.

    Raises:
        ValueError: When scale <= 0.
        TypeError: When the input `dtype` is not a subclass of float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> import mindspore.nn.probability.distribution as msd
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
        >>> class Prob(nn.Cell):
        ...     def __init__(self):
        ...         super(Prob, self).__init__()
        ...         self.gum = msd.Gumbel(np.array([0.0]), np.array([[1.0], [2.0]]), dtype=mindspore.float32)
        ...
        ...     def construct(self, x_):
        ...         return self.gum.prob(x_)
        >>> value = np.array([1.0, 2.0]).astype(np.float32)
        >>> pdf = Prob()
        >>> output = pdf(Tensor(value, dtype=mindspore.float32))
    """

    def __init__(self,
                 loc,
                 scale,
                 seed=0,
                 dtype=mstype.float32,
                 name="Gumbel"):
        """
        Constructor of Gumbel distribution.
        """
        valid_dtype = mstype.float_type
        Validator.check_type_name(
            "dtype", dtype, valid_dtype, type(self).__name__)
        gumbel_cdf = msb.GumbelCDF(loc, scale)
        super(Gumbel, self).__init__(
            distribution=msd.Uniform(0.0, 1.0, dtype=dtype),
            bijector=msb.Invert(gumbel_cdf),
            seed=seed, name=name)

        # overwrite default_parameters and parameter_names
        self._reset_parameters()
        self._loc = self._add_parameter(loc, 'loc')
        self._scale = self._add_parameter(scale, 'scale')
        self._gumbel_bijector = gumbel_cdf

        # ops needed for the class
        self.cast = P.Cast()
        self.const = P.ScalarToTensor()
        self.exp = exp_generic
        self.expm1 = P.Expm1()
        self.fill = P.Fill()
        self.lgamma = nn.LGamma()
        self.log = log_generic
        self.shape = P.Shape()
        self.squeeze = P.Squeeze(0)
        self.sqrt = P.Sqrt()

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

    def extend_repr(self):
        """Display instance object as string."""
        if self.is_scalar_batch:
            str_info = 'loc = {}, scale = {}'.format(self._loc, self._scale)
        else:
            str_info = 'batch_shape = {}'.format(self._broadcast_shape)
        return str_info

    def _get_dist_type(self):
        return "Gumbel"

    def _get_dist_args(self, loc=None, scale=None):
        if scale is None:
            scale = self.scale
        else:
            self.checktensor(scale, 'scale')
        if loc is None:
            loc = self.loc
        else:
            self.checktensor(loc, 'loc')
        return loc, scale

    def _mean(self):
        r"""
        The mean of the distribution.

        .. math::
            MEAN(X) = loc + scale * Euler-Mascheroni_constant
        """
        return self.loc + self.scale * np.euler_gamma

    def _mode(self):
        """
        The mode of the distribution.
        """
        return self.loc * self.fill(self.parameter_type, self.shape(self.scale), 1.0)

    def _sd(self):
        r"""
        The standard deviation of the distribution.

        .. math::
            STD(X) = \frac{\pi}{\sqrt(6)} * scale
        """
        scale = self.scale * \
            self.fill(self.parameter_type, self.broadcast_shape, 1.0)
        return scale * np.pi / self.sqrt(self.const(6., mstype.float32))

    def _entropy(self):
        r"""
        Evaluate entropy.

        .. math::
            H(X) = 1. + \log(scale) + Euler-Mascheroni_constant
        """
        scale = self.scale * \
            self.fill(self.parameter_type, self.broadcast_shape, 1.0)
        return 1. + self.log(scale) + np.euler_gamma

    def _log_prob(self, value):
        r"""
        .. math::
            log_pdf(X) = -(z + \exp(-z)) - \log(scale)
                where z = \frac{x - loc}{scale}
        """
        value = self._check_value(value, 'value')
        value = self.cast(value, self.dtype)
        z = (value - self.loc) / self.scale
        return -(z + self.exp(-z)) - self.log(self.scale)

    def _cdf(self, value):
        r"""
        .. math::
            cdf_pdf(X) = \exp(-\exp(-\frac{x - loc}{scale})
        """
        value = self._check_value(value, 'value')
        value = self.cast(value, self.dtype)
        return self._gumbel_bijector("forward", value)

    def _cross_entropy(self, dist, loc_b, scale_b):
        r"""
        Evaluate cross entropy between Gumbel distributions.

        Args:
            dist (str): The type of the distributions. Should be "Gumbel" in this case.
            loc_b (Tensor): The loc of distribution b.
            scale_b (Tensor): The scale of distribution b.
        """
        check_distribution_name(dist, 'Gumbel')
        return self._entropy() + self._kl_loss(dist, loc_b, scale_b)

    def _kl_loss(self, dist, loc_b, scale_b):
        r"""
        Evaluate Gumbel-Gumbel kl divergence, i.e. KL(a||b).

        Args:
            dist (str): The type of the distributions. Should be "Gumbel" in this case.
            loc_b (Tensor): The loc of distribution b.
            scale_b (Tensor): The scale of distribution b.

        .. math::
            KL(a||b) = \log(scale_b / scale_a) + Euler-Mascheroni_constant * (scale_a / scale_b - 1.) +
                       \exp(\frac{(loc_b - loc_a)}{scale_b}) * \Gamma(scale_a / scale_b + 1.) - 1.
        """
        check_distribution_name(dist, 'Gumbel')
        loc_b = self._check_value(loc_b, 'loc_b')
        scale_b = self._check_value(scale_b, 'scale_b')
        loc_b = self.cast(loc_b, self.parameter_type)
        scale_b = self.cast(scale_b, self.parameter_type)
        return self.log(scale_b / self.scale) +\
            np.euler_gamma * (self.scale / scale_b - 1.) + (self.loc - loc_b) / scale_b +\
            self.expm1((loc_b - self.loc) / scale_b +
                       self.lgamma(self.scale / scale_b + 1.))

    def _sample(self, shape=()):
        shape = self.checktuple(shape, 'shape')
        origin_shape = shape + self._broadcast_shape
        if origin_shape == ():
            sample_shape = (1,)
        else:
            sample_shape = origin_shape
        org_sample = self.distribution("sample", sample_shape)
        org_sample = self.cast(org_sample, self.dtype)
        value = self.bijector("forward", org_sample)
        if origin_shape == ():
            value = self.squeeze(value)
        return value
