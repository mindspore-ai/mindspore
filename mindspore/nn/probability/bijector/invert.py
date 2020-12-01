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

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> import mindspore.nn.probability.bijector as msb
        >>> from mindspore import Tensor
        >>> import mindspore.context as context
        >>> context.set_context(mode=1)
        >>>
        >>> # To initialize an inverse Exp bijector.
        >>> inv_exp = msb.Invert(msb.Exp())
        >>> value = Tensor([1, 2, 3], dtype=mindspore.float32)
        >>> ans1 = inv_exp.forward(value)
        >>> print(ans1)
        [0.        0.6931472 1.0986123]
        >>> ans2 = inv_exp.inverse(value)
        >>> print(ans2)
        [ 2.7182817  7.389056  20.085537 ]
        >>> ans3 = inv_exp.forward_log_jacobian(value)
        >>> print(ans3)
        [-0.        -0.6931472 -1.0986123]
        >>> ans4 = inv_exp.inverse_log_jacobian(value)
        >>> print(ans4)
        [1. 2. 3.]
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
