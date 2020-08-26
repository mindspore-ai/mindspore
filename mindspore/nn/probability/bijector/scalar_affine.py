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
"""Scalar Affine Bijector"""
from mindspore.ops import operations as P
from mindspore._checkparam import Validator as validator
from ..distribution._utils.utils import cast_to_tensor
from ..distribution._utils.custom_ops import log_generic
from .bijector import Bijector


class ScalarAffine(Bijector):
    """
    Scalar Affine Bijector.
    This Bijector performs the operation: Y = a * X + b, where a is the scale
    factor and b is the shift factor.

    Args:
        scale (float): scale factor. Default: 1.0.
        shift (float): shift factor. Default: 0.0.
        name (str): name of the bijector. Default: 'ScalarAffine'.

    Examples:
        >>> # To initialize a ScalarAffine bijector of scale 1 and shift 2
        >>> scalaraffine = nn.probability.bijector.ScalarAffine(1, 2)
        >>>
        >>> # To use ScalarAffine bijector in a network
        >>> class net(Cell):
        >>>     def __init__(self):
        >>>         super(net, self).__init__():
        >>>         self.s1 = nn.probability.bijector.ScalarAffine(1, 2)
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
                 scale=1.0,
                 shift=0.0,
                 name='ScalarAffine'):
        """
        Constructor of scalar affine bijector.
        """
        param = dict(locals())
        validator.check_value_type('scale', scale, [int, float], type(self).__name__)
        validator.check_value_type('shift', shift, [int, float], type(self).__name__)
        self._scale = cast_to_tensor(scale)
        self._shift = cast_to_tensor(shift)
        super(ScalarAffine, self).__init__(
            is_constant_jacobian=True,
            is_injective=True,
            name=name,
            dtype=None,
            param=param)

        self.abs = P.Abs()
        self.oneslike = P.OnesLike()
        self.log = log_generic

    @property
    def scale(self):
        return self._scale

    @property
    def shift(self):
        return self._shift

    def extend_repr(self):
        str_info = f'scale = {self.scale}, shift = {self.shift}'
        return str_info

    def shape_mapping(self, shape):
        return shape

    def _forward(self, x):
        r"""
        .. math::
            f(x) = a * x + b
        """
        x = self._check_value(x, 'value')
        return self.scale * x + self.shift * self.oneslike(x)

    def _inverse(self, y):
        r"""
        .. math::
            f(y) = \frac{y - b}{a}
        """
        y = self._check_value(y, 'value')
        return (y - self.shift) / self.scale

    def _forward_log_jacobian(self, x):
        r"""
        .. math::
            f(x) = a * x + b
            f'(x) = a
            \log(f'(x)) = \log(a)
        """
        x = self._check_value(x, 'value')
        return self.log(self.abs(self.scale))

    def _inverse_log_jacobian(self, y):
        r"""
        .. math::
            f(y) = \frac{(y - b)}{a}
            f'(x) = \frac{1.0}{a}
            \log(f'(x)) = - \log(a)
        """
        y = self._check_value(y, 'value')
        return -1. * self.log(self.abs(self.scale))
