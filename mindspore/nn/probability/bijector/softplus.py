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
"""Softplus Bijector"""
import numpy as np
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindspore.nn.layer.activation import LogSigmoid
from mindspore._checkparam import Validator as validator
from ..distribution._utils.utils import cast_to_tensor
from ..distribution._utils.custom_ops import exp_generic, expm1_generic, log_generic
from .bijector import Bijector


class Softplus(Bijector):
    r"""
    Softplus Bijector.
    This Bijector performs the operation, where k is the sharpness factor.

    .. math::
        Y = \frac{\log(1 + e ^ {kX})}{k}

    Args:
        sharpness (float): scale factor. Default: 1.0.
        name (str): name of the bijector. Default: 'Softplus'.

    Examples:
        >>> # To initialize a Softplus bijector of sharpness 2
        >>> softplus = nn.probability.bijector.Softfplus(2)
        >>>
        >>> # To use ScalarAffine bijector in a network
        >>> class net(Cell):
        >>>     def __init__(self):
        >>>         super(net, self).__init__():
        >>>         self.sp1 = nn.probability.bijector.Softflus(2)
        >>>
        >>>     def construct(self, value):
        >>>         # Similar calls can be made to other probability functions
        >>>         # by replacing 'forward' with the name of the function
        >>>         ans1 = self.sp1.forward(value)
        >>>         ans2 = self.sp1.inverse(value)
        >>>         ans3 = self.sp1.forward_log_jacobian(value)
        >>>         ans4 = self.sp1.inverse_log_jacobian(value)
    """

    def __init__(self,
                 sharpness=1.0,
                 name='Softplus'):
        param = dict(locals())
        validator.check_value_type('sharpness', sharpness, [int, float], type(self).__name__)
        super(Softplus, self).__init__(name=name, param=param)
        self._sharpness = cast_to_tensor(sharpness)

        self.exp = exp_generic
        self.log = log_generic
        self.expm1 = expm1_generic
        self.abs = P.Abs()
        self.fill = P.Fill()
        self.greater = P.Greater()
        self.less = P.Less()
        self.log_sigmoid = LogSigmoid()
        self.logicalor = P.LogicalOr()
        self.select = P.Select()
        self.shape = P.Shape()
        self.sigmoid = P.Sigmoid()
        self.softplus = self._softplus
        self.inverse_softplus = self._inverse_softplus

        self.threshold = np.log(np.finfo(np.float32).eps) + 1
        self.tiny = np.exp(self.threshold)

    def _softplus(self, x):
        too_small = self.less(x, self.threshold)
        too_large = self.greater(x, -self.threshold)
        too_small_value = self.exp(x)
        too_large_value = x
        ones = self.fill(mstype.float32, self.shape(x), 1.0)
        too_small_or_too_large = self.logicalor(too_small, too_large)
        x = self.select(too_small_or_too_large, ones, x)
        y = self.log(self.exp(x) + 1.0)
        return self.select(too_small, too_small_value, self.select(too_large, too_large_value, y))

    def _inverse_softplus(self, x):
        r"""
        .. math::
            f(x) = \frac{\log(1 + e^{x}))}
            f^{-1}(y) = \frac{\log(e^{y} - 1)}
        """
        too_small = self.less(x, self.tiny)
        too_large = self.greater(x, -self.threshold)
        too_small_value = self.log(x)
        too_large_value = x
        ones = self.fill(mstype.float32, self.shape(x), 1.0)
        too_small_or_too_large = self.logicalor(too_small, too_large)
        x = self.select(too_small_or_too_large, ones, x)
        y = x + self.log(self.abs(self.expm1(-x)))
        return self.select(too_small, too_small_value, self.select(too_large, too_large_value, y))

    @property
    def sharpness(self):
        return self._sharpness

    def extend_repr(self):
        str_info = f'sharpness = {self.sharpness}'
        return str_info

    def shape_mapping(self, shape):
        return shape

    def _forward(self, x):
        x = self._check_value(x, 'value')
        scaled_value = self.sharpness * x
        return self.softplus(scaled_value) / self.sharpness

    def _inverse(self, y):
        r"""
        .. math::
            f(x) = \frac{\log(1 + e^{kx}))}{k}
            f^{-1}(y) = \frac{\log(e^{ky} - 1)}{k}
        """
        y = self._check_value(y, 'value')
        scaled_value = self.sharpness * y
        return self.inverse_softplus(scaled_value) / self.sharpness

    def _forward_log_jacobian(self, x):
        r"""
        .. math:
            f(x) = \log(1 + e^{kx}) / k
            f'(x) = \frac{e^{kx}}{ 1 + e^{kx}}
            \log(f'(x)) =  kx - \log(1 + e^{kx}) = kx - f(kx)
        """
        x = self._check_value(x, 'value')
        scaled_value = self.sharpness * x
        return self.log_sigmoid(scaled_value)

    def _inverse_log_jacobian(self, y):
        r"""
        .. math:
            f(y) = \frac{\log(e^{ky} - 1)}{k}
            f'(y) = \frac{e^{ky}}{e^{ky} - 1}
            \log(f'(y)) = ky - \log(e^{ky} - 1) = ky - f(ky)
        """
        y = self._check_value(y, 'value')
        scaled_value = self.sharpness * y
        return scaled_value - self.inverse_softplus(scaled_value)
