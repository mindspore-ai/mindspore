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

    Note:
        For `inverse` and `inverse_log_jacobian`, input should be in range of (0, 1).

    Args:
        loc (float, list, numpy.ndarray, Tensor): The location. Default: 0..
        scale (float, list, numpy.ndarray, Tensor): The scale. Default: 1.0.
        name (str): The name of the Bijector. Default: 'Gumbel_CDF'.

    Examples:
        >>> # To initialize a GumbelCDF bijector of loc 0.0, and scale 1.0.
        >>> import mindspore.nn.probability.bijector as msb
        >>> gum = msb.GumbelCDF(0.0, 1.0)
        >>>
        >>> # To use GumbelCDF bijector in a network.
        >>> class net(Cell):
        ...     def __init__(self):
        ...         super(net, self).__init__():
        ...         self.gum = msb.GumbelCDF(0.0, 1.0)
        ...
        ...     def construct(self, value):
        ...         # Similar calls can be made to other functions
        ...         # by replacing 'forward' by the name of the function.
        ...         ans1 = self.gum.forward(value)
        ...         ans2 = self.gum.inverse(value)
        ...         ans3 = self.gum.forward_log_jacobian(value)
        ...         ans4 = self.gum.inverse_log_jacobian(value)
        ...
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
        return self._loc

    @property
    def scale(self):
        return self._scale

    def extend_repr(self):
        if self.is_scalar_batch:
            str_info = f'loc = {self.loc}, scale = {self.scale}'
        else:
            str_info = f'batch_shape = {self.batch_shape}'
        return str_info

    def _forward(self, x):
        x = self._check_value_dtype(x)
        loc_local = self.cast_param_by_value(x, self.loc)
        scale_local = self.cast_param_by_value(x, self.scale)
        z = (x - loc_local) / scale_local
        return self.exp(-self.exp(-z))

    def _inverse(self, y):
        y = self._check_value_dtype(y)
        loc_local = self.cast_param_by_value(y, self.loc)
        scale_local = self.cast_param_by_value(y, self.scale)
        return loc_local - scale_local * self.log(-self.log(y))

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
