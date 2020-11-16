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
"""Invert Bijector"""
from mindspore._checkparam import Validator as validator
from .bijector import Bijector


class Invert(Bijector):
    r"""
    Invert Bijector.

    Args:
        bijector (Bijector): Base Bijector.
        name (str): The name of the Bijector. Default: Invert.

    Examples:
        >>> # To initialize an Invert bijector.
        >>> import mindspore.nn.probability.bijector as msb
        >>> n = msb.Invert(msb.Exp())
        >>>
        >>> # To use an Invert bijector in a network.
        >>> class net(Cell):
        ...     def __init__(self):
        ...         super(net, self).__init__():
        ...         self.inv = msb.Invert(msb.Exp())
        ...
        ...     def construct(self, value):
        ...         # Similar calls can be made to other functions
        ...         # by replacing `forward` by the name of the function.
        ...         ans1 = self.inv.forward(value)
        ...         ans2 = self.inv.inverse(value)
        ...         ans3 = self.inv.forward_log_jacobian(value)
        ...         ans4 = self.inv.inverse_log_jacobian(value)
    """

    def __init__(self,
                 bijector,
                 name='Invert'):
        param = dict(locals())
        validator.check_value_type('bijector', bijector, [Bijector], "Invert")
        name = (name + bijector.name) if name == 'Invert' else name
        super(Invert, self).__init__(is_constant_jacobian=bijector.is_constant_jacobian,
                                     is_injective=bijector.is_injective,
                                     name=name,
                                     dtype=bijector.dtype,
                                     param=param)
        self._bijector = bijector
        self._batch_shape = self.bijector.batch_shape
        self._is_scalar_batch = self.bijector.is_scalar_batch

    @property
    def bijector(self):
        return self._bijector

    def inverse(self, y):
        return self.bijector("forward", y)

    def forward(self, x):
        return self.bijector("inverse", x)

    def inverse_log_jacobian(self, y):
        return self.bijector("forward_log_jacobian", y)

    def forward_log_jacobian(self, x):
        return self.bijector("inverse_log_jacobian", x)
