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
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from ..distribution._utils.custom_ops import exp_generic, expm1_generic, log_generic, log1p_generic
from .bijector import Bijector


class PowerTransform(Bijector):
    r"""
    Power Bijector.
    This Bijector performs the operation: Y = g(X) = (1 + X * c)^(1 / c), X >= -1 / c, where c >= 0 is the power.

    The power transform maps inputs from `[-1/c, inf]` to `[0, inf]`.

    This bijector is equivalent to the `Exp` bijector when `c=0`

    Raises:
        ValueError: If the power is less than 0 or is not known statically.

    Args:
        power (int or float): scale factor. Default: 0.
        name (str): name of the bijector. Default: 'PowerTransform'.

    Examples:
        >>> # To initialize a PowerTransform bijector of power 0.5
        >>> import mindspore.nn.probability.bijector as msb
        >>> n = msb.PowerTransform(0.5)
        >>>
        >>> # To use PowerTransform distribution in a network
        >>> class net(Cell):
        >>>     def __init__(self):
        >>>         super(net, self).__init__():
        >>>         self.p1 = msb.PowerTransform(0.5)
        >>>
        >>>     def construct(self, value):
        >>>         # Similar calls can be made to other probability functions
        >>>         # by replacing 'forward' with the name of the function
        >>>         ans1 = self.s1.forward(value)
        >>>         ans2 = self.s1.inverse(value)
        >>>         ans3 = self.s1.forward_log_jacobian(value)
        >>>         ans4 = self.s1.inverse_log_jacobian(value)
    """

    def __init__(self,
                 power=0,
                 name='PowerTransform',
                 param=None):
        param = dict(locals()) if param is None else param
        super(PowerTransform, self).__init__(name=name, param=param)
        validator.check_value_type('power', power, [int, float], self.name)
        validator.check_number("power", power, 0, Rel.GE, self.name)
        self._power = power
        self.pow = P.Pow()
        self.exp = exp_generic
        self.expm1 = expm1_generic
        self.log = log_generic
        self.log1p = log1p_generic

    @property
    def power(self):
        return self._power

    def extend_repr(self):
        str_info = f'power = {self.power}'
        return str_info

    def shape_mapping(self, shape):
        return shape

    def _forward(self, x):
        x = self._check_value(x, 'value')
        if self.power == 0:
            return self.exp(x)
        return self.exp(self.log1p(x * self.power) / self.power)

    def _inverse(self, y):
        y = self._check_value(y, 'value')
        if self.power == 0:
            return self.log(y)
        return self.expm1(self.log(y) * self.power) / self.power

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
        x = self._check_value(x, 'value')
        if self.power == 0:
            return x
        return (1. / self.power - 1) * self.log1p(x * self.power)

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
        y = self._check_value(y, 'value')
        return (self.power - 1) * self.log(y)
