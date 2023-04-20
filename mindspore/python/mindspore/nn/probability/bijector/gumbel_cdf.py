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
"""GumbelCDF Bijector"""
from mindspore.ops import operations as P
from ..distribution._utils.utils import check_greater_zero
from ..distribution._utils.custom_ops import exp_generic, log_generic
from .bijector import Bijector


class GumbelCDF(Bijector):
    r"""
    GumbelCDF Bijector.
    This Bijector performs the operation:

    .. math::
        Y = \exp(-\exp(\frac{-(X - loc)}{scale}))

    Args:
        loc (float, list, numpy.ndarray, Tensor): The location. Default: ``0.0`` .
        scale (float, list, numpy.ndarray, Tensor): The scale. Default: ``1.0`` .
        name (str): The name of the Bijector. Default: ``'GumbelCDF'`` .

    Note:
        `scale` must be greater than zero.
        For `inverse` and `inverse_log_jacobian`, input should be in range of (0, 1).
        The dtype of `loc` and `scale` must be float.
        If `loc`, `scale` are passed in as numpy.ndarray or tensor, they have to have
        the same dtype otherwise an error will be raised.

    Raises:
        TypeError: When the dtype of `loc` or `scale` is not float,
                   or when the dtype of `loc` and `scale` is not same.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> import mindspore.nn.probability.bijector as msb
        >>> from mindspore import Tensor
        >>>
        >>> # To initialize a GumbelCDF bijector of loc 1.0, and scale 2.0.
        >>> gumbel_cdf = msb.GumbelCDF(1.0, 2.0)
        >>> # To use a GumbelCDF bijector in a network.
        >>> x = Tensor([1, 2, 3], dtype=mindspore.float32)
        >>> y = Tensor([0.1, 0.2, 0.3], dtype=mindspore.float32)
        >>> ans1 = gumbel_cdf.forward(x)
        >>> print(ans1.shape)
        (3,)
        >>> ans2 = gumbel_cdf.inverse(y)
        >>> print(ans2.shape)
        (3,)
        >>> ans3 = gumbel_cdf.forward_log_jacobian(x)
        >>> print(ans3.shape)
        (3,)
        >>> ans4 = gumbel_cdf.inverse_log_jacobian(y)
        >>> print(ans4.shape)
        (3,)
    """

    def __init__(self,
                 loc=0.0,
                 scale=1.0,
                 name='GumbelCDF'):
        """
        Constructor of GumbelCDF Bijector.
        """
        param = dict(locals())
        param['param_dict'] = {'loc': loc, 'scale': scale}
        super(GumbelCDF, self).__init__(name=name, param=param)

        self._loc = self._add_parameter(loc, 'loc')
        self._scale = self._add_parameter(scale, 'scale')
        check_greater_zero(self._scale, "scale")

        self.cast = P.Cast()
        self.exp = exp_generic
        self.log = log_generic

    @property
    def loc(self):
        """
        Return the loc parameter of the bijector.

        Output:
            Tensor, the loc parameter of the bijector.
        """
        return self._loc

    @property
    def scale(self):
        """
        Return the scale parameter of the bijector.

        Output:
            Tensor, the scale parameter of the bijector.
        """
        return self._scale

    def extend_repr(self):
        """Display instance object as string."""
        if self.is_scalar_batch:
            str_info = 'loc = {}, scale = {}'.format(self.loc, self.scale)
        else:
            str_info = 'batch_shape = {}'.format(self.batch_shape)
        return str_info

    def _forward(self, x):
        x = self._check_value_dtype(x)
        loc_local = self.cast_param_by_value(x, self.loc)
        scale_local = self.cast_param_by_value(x, self.scale)
        z = (x - loc_local) / scale_local
        return self.exp((-1) * self.exp(-z))

    def _inverse(self, y):
        y = self._check_value_dtype(y)
        loc_local = self.cast_param_by_value(y, self.loc)
        scale_local = self.cast_param_by_value(y, self.scale)
        return loc_local - scale_local * self.log((-1) * self.log(y))

    def _forward_log_jacobian(self, x):
        x = self._check_value_dtype(x)
        loc_local = self.cast_param_by_value(x, self.loc)
        scale_local = self.cast_param_by_value(x, self.scale)
        z = (x - loc_local) / scale_local
        return -z - self.exp(-z) - self.log(scale_local)

    def _inverse_log_jacobian(self, y):
        y = self._check_value_dtype(y)
        scale_local = self.cast_param_by_value(y, self.scale)
        return self.log(scale_local / (-1. * y * self.log(y)))
