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
"""Power Bijector"""
from mindspore.ops import operations as P
from ..distribution._utils.utils import check_greater_equal_zero
from ..distribution._utils.custom_ops import exp_generic, expm1_generic, log_generic, log1p_generic
from .bijector import Bijector


class PowerTransform(Bijector):
    r"""
    Power Bijector.
    This Bijector performs the operation:

    .. math::
        Y = g(X) = (1 + X * c)^{1 / c}, X >= -1 / c
    where c >= 0 is the power.

    The power transform maps inputs from `[-1/c, inf]` to `[0, inf]`.

    This Bijector is equivalent to the `Exp` bijector when `c=0`.

    Raises:
        ValueError: When the power is less than 0 or is not known statically.

    Args:
        power (float, list, numpy.ndarray, Tensor): The scale factor. Default: 0.
        name (str): The name of the bijector. Default: 'PowerTransform'.

    Examples:
        >>> # To initialize a PowerTransform bijector of power 0.5.
        >>> import mindspore.nn.probability.bijector as msb
        >>> n = msb.PowerTransform(0.5)
        >>>
        >>> # To use a PowerTransform bijector in a network.
        >>> class net(Cell):
        ...     def __init__(self):
        ...         super(net, self).__init__():
        ...         self.p1 = msb.PowerTransform(0.5)
        ...
        ...     def construct(self, value):
        ...         # Similar calls can be made to other functions
        ...         # by replacing 'forward' by the name of the function.
        ...         ans1 = self.s1.forward(value)
        ...         ans2 = self.s1.inverse(value)
        ...         ans3 = self.s1.forward_log_jacobian(value)
        ...         ans4 = self.s1.inverse_log_jacobian(value)
        ...
    """

    def __init__(self,
                 power=0.,
                 name='PowerTransform'):
        param = dict(locals())
        param['param_dict'] = {'power': power}
        super(PowerTransform, self).__init__(name=name, param=param)
        self._power = self._add_parameter(power, 'power')
        check_greater_equal_zero(self._power, 'Power')

        self.pow = P.Pow()
        self.dtypeop = P.DType()
        self.cast = P.Cast()
        self.exp = exp_generic
        self.expm1 = expm1_generic
        self.log = log_generic
        self.log1p = log1p_generic

    @property
    def power(self):
        return self._power

    def extend_repr(self):
        if self.is_scalar_batch:
            str_info = f'power = {self.power}'
        else:
            str_info = f'batch_shape = {self.batch_shape}'
        return str_info


    def _forward(self, x):
        x = self._check_value_dtype(x)
        power_local = self.cast_param_by_value(x, self.power)
        if power_local == 0:
            forward_v = self.exp(x)
        else:
            forward_v = self.exp(self.log1p(x * power_local) / power_local)
        return forward_v

    def _inverse(self, y):
        y = self._check_value_dtype(y)
        power_local = self.cast_param_by_value(y, self.power)
        if power_local == 0:
            inverse_v = self.log(y)
        else:
            inverse_v = self.expm1(self.log(y) * power_local) / power_local
        return inverse_v

    def _forward_log_jacobian(self, x):
        r"""
        .. math:
            if c == 0:
                f(x) = e^x
                f'(x) = e^x
                \log(f'(x)) = \log(e^x) = x
            else:
                f(x) = e^\frac{\log(xc + 1)}{c}
                f'(x) = e^\frac{\log(xc + 1)}{c} * \frac{1}{xc + 1}
                \log(f'(x)) =  (\frac{1}{c} - 1) * \log(xc + 1)
        """
        x = self._check_value_dtype(x)
        power_local = self.cast_param_by_value(x, self.power)
        if power_local == 0:
            forward_log_j = x
        else:
            forward_log_j = (1. / power_local - 1) * self.log1p(x * power_local)
        return forward_log_j

    def _inverse_log_jacobian(self, y):
        r"""
        .. math:
            if c == 0:
                f(x) = \log(x)
                f'(x) = \frac{1}{x}
                \log(f'(x)) = \log(\frac{1}{x}) = -\log(x)
            else:
                f(x) = \frac{e^\log(y)*c + 1}{c}
                f'(x) = \frac{e^c\log(y)}{y}
                \log(f'(x)) =  \log(\frac{e^c\log(y)}{y}) = (c-1) * \log(y)
        """
        y = self._check_value_dtype(y)
        power_local = self.cast_param_by_value(y, self.power)
        inverse_log_j = (power_local - 1) * self.log(y)
        return inverse_log_j
