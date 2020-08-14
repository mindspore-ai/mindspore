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
from mindspore.ops import operations as P
from mindspore.nn.layer.activation import LogSigmoid
from mindspore._checkparam import Validator as validator
from ..distribution._utils.utils import cast_to_tensor
from .bijector import Bijector

class Softplus(Bijector):
    r"""
    Softplus Bijector.
    This Bijector performs the operation, where k is the sharpness factor.

    .. math::
    Y = \frac{\log(1 + e ^ {kX})}{k}

    Args:
        sharpness (float): scale factor. Default: 1.0.

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
        >>>         ans = self.sp1.forward(value)
        >>>         ans = self.sp1.inverse(value)
        >>>         ans = self.sp1.forward_log_jacobian(value)
        >>>         ans = self.sp1.inverse_log_jacobian(value)
    """
    def __init__(self,
                 sharpness=1.0,
                 name='Softplus'):
        param = dict(locals())
        validator.check_value_type('sharpness', sharpness, [float], name)
        super(Softplus, self).__init__(name=name, param=param)
        self._sharpness = cast_to_tensor(sharpness)

        self.exp = P.Exp()
        self.expm1 = self._expm1_by_step
        self.log_sigmoid = LogSigmoid()
        self.log = P.Log()
        self.sigmoid = P.Sigmoid()

        self.softplus = self._softplus
        self.inverse_softplus = self._inverse_softplus

    def _expm1_by_step(self, x):
        """
        Expm1 ops under GPU context.
        """
        return self.exp(x) - 1.0

    def _softplus(self, x):
        return self.log(self.exp(x) + 1.0)

    def _inverse_softplus(self, x):
        r"""
        .. math::
            f(x) = \frac{\log(1 + e^{x}))}
            f^{-1}(y) = \frac{\log(e^{y} - 1)}
        """
        return self.log(self.expm1(x))

    @property
    def sharpness(self):
        return self._sharpness

    def extend_repr(self):
        str_info = f'sharpness = {self.sharpness}'
        return str_info

    def shape_mapping(self, shape):
        return shape

    def _forward(self, x):
        scaled_value = self.sharpness * x
        return self.softplus(scaled_value) / self.sharpness

    def _inverse(self, y):
        r"""
        .. math::
            f(x) = \frac{\log(1 + e^{kx}))}{k}
            f^{-1}(y) = \frac{\log(e^{ky} - 1)}{k}
        """
        scaled_value = self.sharpness * y
        return self.inverse_softplus(scaled_value) / self.sharpness

    def _forward_log_jacobian(self, x):
        r"""
        .. math:
            f(x) = \log(1 + e^{kx}) / k
            f'(x) = \frac{e^{kx}}{ 1 + e^{kx}}
            \log(f'(x)) =  kx - \log(1 + e^{kx}) = kx - f(kx)
        """
        scaled_value = self.sharpness * x
        return self.log_sigmoid(scaled_value)

    def _inverse_log_jacobian(self, y):
        r"""
        .. math:
            f(y) = \frac{\log(e^{ky} - 1)}{k}
            f'(y) = \frac{e^{ky}}{e^{ky} - 1}
            \log(f'(y)) = ky - \log(e^{ky} - 1) = ky - f(ky)
        """
        scaled_value = self.sharpness * y
        return scaled_value - self.inverse_softplus(scaled_value)
